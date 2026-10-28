"""Where are GPUs likely free right now? Estimated without submitting anything.

    python -m nrp_kit.capacity [--models "A10|L4|RTX 4090"] [--min 1]

A tenant usually cannot list other namespaces' pods, but can list nodes, and the cluster's GPU
metrics are published through its Grafana. The DCGM exporter emits one series per GPU that is
attached to a pod, labelled with the pod's Hostname. So for each schedulable, untainted GPU node:

    likely free = allocatable GPUs - GPUs with an exporter series on that host

A node with no series at all is either idle or not covered by the exporter; such nodes are
reported as "no series", not as known free. One pinned submission settles which (the scheduler
answers within seconds), so pick from the list and submit one Job, not several.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import urllib.request
from dataclasses import dataclass

from . import config, kube


@dataclass(frozen=True)
class NodeGPUs:
    node: str
    zone: str
    product: str
    allocatable: int
    attached: int  # GPUs with an exporter series (attached to some pod)
    has_series: bool

    @property
    def likely_free(self) -> int:
        return max(0, self.allocatable - self.attached)


def prom(site, expr: str) -> list[tuple[dict, float]]:
    if not site.grafana_url or not site.grafana_ds:
        raise SystemExit("set grafana_url and grafana_ds (datasource uid) in the site config")
    body = json.dumps({"queries": [{"refId": "A", "datasource": {"type": "prometheus", "uid": site.grafana_ds},
                                    "expr": expr, "instant": True}], "from": "now-5m", "to": "now"}).encode()
    req = urllib.request.Request(site.grafana_url.rstrip("/") + "/api/ds/query", body,
                                 {"Content-Type": "application/json", "User-Agent": "nrp-kit"})
    j = json.load(urllib.request.urlopen(req, timeout=60))
    return [(f["schema"]["fields"][1].get("labels", {}), f["data"]["values"][1][0])
            for f in j["results"]["A"].get("frames", []) if len(f["schema"]["fields"]) > 1]


def gpu_nodes(site) -> list[NodeGPUs]:
    attached = {lab.get("Hostname"): int(v) for lab, v in prom(site, "count by (Hostname) (DCGM_FI_DEV_GPU_UTIL)")}
    r = kube.kubectl(site, "get", "nodes", "-o", "json")
    if r.returncode:
        raise SystemExit(f"cannot list nodes: {r.stderr.strip()}")
    out = []
    for n in json.loads(r.stdout)["items"]:
        lab = n["metadata"]["labels"]
        prod = lab.get("nvidia.com/gpu.product")
        if not prod or n["spec"].get("unschedulable"):
            continue
        if any(t["key"] != "nvidia.com/gpu" for t in n["spec"].get("taints", [])):
            continue  # reserved, testing, unreachable or otherwise not for general use
        name = n["metadata"]["name"]
        alloc = int((n["status"].get("allocatable", {}) or {}).get("nvidia.com/gpu", "0") or 0)
        out.append(NodeGPUs(name, lab.get("topology.kubernetes.io/zone", "?"), prod, alloc,
                            attached.get(name, 0), name in attached))
    return out


def summary(nodes: list[NodeGPUs], models: str | None = None) -> list[tuple[str, str, int, int, int]]:
    """(zone, product, likely free on monitored nodes, GPUs on nodes with no series, nodes) rows."""
    rx = re.compile(models) if models else None
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for n in nodes:
        if rx and not rx.search(n.product):
            continue
        a = agg[(n.zone, n.product)]
        if n.has_series:
            a[0] += n.likely_free
        else:
            a[1] += n.allocatable
        a[2] += 1
    return sorted(((z, p, *v) for (z, p), v in agg.items()), key=lambda r: (-r[2], -r[3]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="regex on the GPU product label, e.g. 'A10$|L4$|RTX-4090'")
    ap.add_argument("--min", type=int, default=1, help="hide rows with fewer free (or unseen) GPUs")
    a = ap.parse_args(argv)
    site = config.load()
    print(f"{'zone':20s} {'product':34s} {'free':>5s} {'no-series':>9s} {'nodes':>5s}")
    for z, p, free, unseen, k in summary(gpu_nodes(site), a.models):
        if free + unseen >= a.min:
            print(f"{z:20s} {p:34s} {free:5d} {unseen:9d} {k:5d}")


if __name__ == "__main__":
    main()
