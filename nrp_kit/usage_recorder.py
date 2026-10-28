"""Record what our own pods use, and optionally stop ours that hold more than they use.

    python -m nrp_kit.usage_recorder --selector app=myrun --observations obs.json --heartbeat hb

Runs on the submitting host, not in the cluster. Every --interval seconds it reads the metrics
API for the namespace, keeps a rolling window per pod matching --selector, and:

- records each Job's mean CPU, mean memory and peak memory into --observations as soon as two
  samples exist, so even a short pilot leaves a measurement that sizing.py can size its class
  from (peaks only grow, so a later quiet window cannot erase one);
- once a pod is older than --grace with a full window, reports any pod whose mean sits below
  the low end of the cluster's usage band for its request; with --stop-underused it also saves
  the Job's log and deletes the Job, returning the capacity to other users.

It never touches pods it did not match. A snapshot of `kubectl top` says little about a pod
whose phases differ (a long read, then compute), which is why it keeps a window and judges the
mean. --heartbeat is touched every cycle, so a submitter can require a live recorder before it
sends the first pod of a class.
"""

from __future__ import annotations

import argparse
import calendar
import collections
import json
import os
import time

from . import config, kube

RECORD_MIN_SAMPLES = 2


def parse_cpu(v: str) -> float:
    v = v.strip()
    for suf, mul in (("n", 1e-9), ("u", 1e-6), ("m", 1e-3)):
        if v.endswith(suf):
            return float(v[:-1]) * mul
    return float(v)


def parse_mem(v: str) -> float:
    """Kubernetes memory quantity -> bytes."""
    v = v.strip()
    for suf, mul in (("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("K", 1e3), ("M", 1e6),
                     ("G", 1e9), ("m", 1e-3)):
        if v.endswith(suf):
            return float(v[: -len(suf)]) * mul
    return float(v)


def requests(site, selector: str) -> dict:
    """{pod: (cpu cores, memory bytes, job, age s)} for Running pods matching the selector."""
    r = kube.kubectl(site, "get", "pods", "-l", selector, "--field-selector=status.phase=Running", "-o", "json")
    if r.returncode:
        return {}
    now, out = time.time(), {}
    for p in json.loads(r.stdout).get("items", []):
        res = p["spec"]["containers"][0].get("resources", {}).get("requests", {})
        if not res.get("cpu") or not res.get("memory"):
            continue
        job = next((o["name"] for o in p["metadata"].get("ownerReferences", []) if o["kind"] == "Job"), None)
        start = p["status"].get("startTime")
        age = now - calendar.timegm(time.strptime(start, "%Y-%m-%dT%H:%M:%SZ")) if start else 0.0
        out[p["metadata"]["name"]] = (parse_cpu(res["cpu"]), parse_mem(res["memory"]), job, age)
    return out


def parse_metrics(doc: dict) -> dict:
    """{pod: (cores, bytes)} from a metrics.k8s.io PodMetricsList, summed over containers."""
    out = {}
    for p in doc.get("items", []):
        cs = p.get("containers", [])
        out[p["metadata"]["name"]] = (sum(parse_cpu(c["usage"]["cpu"]) for c in cs),
                                      sum(parse_mem(c["usage"]["memory"]) for c in cs))
    return out


def usage(site) -> dict:
    """Read the metrics API directly: `kubectl top` fails the whole namespace when any one pod
    has no metrics yet, and a recorder that sees nothing records nothing."""
    r = kube.kubectl(site, "get", "--raw", f"/apis/metrics.k8s.io/v1beta1/namespaces/{site.namespace}/pods")
    try:
        return parse_metrics(json.loads(r.stdout))
    except (ValueError, KeyError):
        return {}


def assess(h, age, req_cpu, req_mem, *, grace, window, floor, small_cpu, small_mem):
    """(measured, low): measured = (mean cores, mean bytes, peak bytes) once RECORD_MIN_SAMPLES
    exist; low = reasons the mean sits under ``floor`` of the request, only after ``grace``."""
    if len(h) < RECORD_MIN_SAMPLES:
        return None, []
    mc = sum(x[0] for x in h) / len(h)
    mm = sum(x[1] for x in h) / len(h)
    measured = (mc, mm, max(x[1] for x in h))
    if age < grace or len(h) < window or (req_cpu <= small_cpu and req_mem <= small_mem):
        return measured, []
    low = []
    if mc < floor * req_cpu:
        low.append(f"cpu {mc:.2f}/{req_cpu:g} cores = {100 * mc / req_cpu:.0f}%")
    if mm < floor * req_mem:
        low.append(f"mem {mm / 2**30:.1f}/{req_mem / 2**30:.1f} GiB = {100 * mm / req_mem:.0f}%")
    return measured, low


def record(path, job, mean_cpu, mean_mem, peak_mem, req_cpu, req_mem, samples):
    try:
        all_obs = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        all_obs = {}
    prev = all_obs.get(job, {})
    all_obs[job] = dict(
        mean_cpu_cores=round(mean_cpu, 3), mean_mem_gib=round(mean_mem / 2**30, 3),
        peak_mem_gib=round(max(peak_mem / 2**30, prev.get("peak_mem_gib", 0)), 3),
        request_cpu=req_cpu, request_mem_gib=round(req_mem / 2**30, 3), samples=samples,
        updated=time.strftime("%FT%TZ", time.gmtime()))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(all_obs, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--window", type=int, default=10, help="samples kept per pod")
    ap.add_argument("--grace", type=float, default=900.0, help="seconds before a pod is judged")
    ap.add_argument("--observations", help="JSON file of per-Job measured usage")
    ap.add_argument("--heartbeat", help="touched every cycle")
    ap.add_argument("--stop-underused", action="store_true", help="delete our Jobs whose mean sits under the band")
    ap.add_argument("--log-dir", help="save a stopped Job's log here first")
    ap.add_argument("--max-cycles", type=int, default=0, help="stop after this many cycles (0: run until interrupted)")
    a = ap.parse_args(argv)
    site = config.load()
    pol = site.policy
    hist = collections.defaultdict(lambda: collections.deque(maxlen=a.window))
    acted, cycles = set(), 0
    while not a.max_cycles or cycles < a.max_cycles:
        cycles += 1
        if a.heartbeat:
            with open(a.heartbeat, "w") as fh:
                fh.write(time.strftime("%FT%TZ", time.gmtime()))
        req, use = requests(site, a.selector), usage(site)
        for pod, (rc, rm, job, age) in sorted(req.items()):
            if pod in use:
                hist[pod].append(use[pod])
            if job in acted:
                continue
            measured, low = assess(hist[pod], age, rc, rm, grace=a.grace, window=a.window,
                                   floor=min(pol.cpu_band[0], pol.memory_band[0]),
                                   small_cpu=pol.small_class_cpu, small_mem=pol.small_class_mem_gib * 2**30)
            if measured and a.observations and job:
                record(a.observations, job, *measured, rc, rm, len(hist[pod]))
            if not low:
                continue
            stamp = time.strftime("%FT%TZ", time.gmtime())
            print(f"{stamp} UNDER-USED {pod} job={job} " + "; ".join(low), flush=True)
            if a.stop_underused and job:
                if a.log_dir:
                    os.makedirs(a.log_dir, exist_ok=True)
                    with open(os.path.join(a.log_dir, f"{job}.log"), "w", encoding="utf-8") as fh:
                        fh.write(f"# stopped {stamp}: " + "; ".join(low) + "\n" + kube.logs(site, job))
                r = kube.kubectl(site, "delete", "job", job)
                print(f"{stamp} {'STOPPED' if not r.returncode else 'STOP FAILED'} {job}", flush=True)
                acted.add(job)
        for pod in list(hist):
            if pod not in req:
                del hist[pod]
        time.sleep(a.interval)  # on the submitting host


if __name__ == "__main__":
    main()
