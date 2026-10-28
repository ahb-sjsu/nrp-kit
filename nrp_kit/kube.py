"""Thin kubectl wrappers: job state, active jobs, GPU pods in the namespace."""

from __future__ import annotations

import json
import subprocess

from .config import Site


def kubectl(site: Site, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    cmd = [site.kubectl, "-n", site.namespace, f"--request-timeout={site.request_timeout_s}s", *args]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=2 * site.request_timeout_s,
                              input=stdin)
    except subprocess.TimeoutExpired:  # an API stall is a missed read, not a crash
        return subprocess.CompletedProcess(cmd, 124, "", "kubectl timed out")


def job_state(site: Site, name: str) -> str:
    """'absent', 'active' or 'finished'."""
    r = kubectl(site, "get", "job", name, "-o", "json")
    if r.returncode != 0:
        return "absent"
    st = json.loads(r.stdout).get("status", {})
    if st.get("succeeded") or st.get("failed") or any(
        c.get("type") in ("Complete", "Failed") and c.get("status") == "True" for c in st.get("conditions", [])
    ):
        return "finished"
    return "active"


def job_names(site: Site, selector: str) -> list[str]:
    r = kubectl(site, "get", "jobs", "-l", selector, "-o", "jsonpath={.items[*].metadata.name}")
    return r.stdout.split() if r.returncode == 0 else []


def active_jobs(site: Site, selector: str) -> list[str]:
    return [n for n in job_names(site, selector) if job_state(site, n) == "active"]


def gpu_pods(site: Site) -> int:
    """GPUs requested by Pending or Running pods in the namespace (all users of it)."""
    r = kubectl(site, "get", "pods", "-o", "json")
    if r.returncode != 0:
        raise SystemExit(f"cannot list pods: {r.stderr.strip()}")
    n = 0
    for p in json.loads(r.stdout).get("items", []):
        if p["status"].get("phase") not in ("Pending", "Running"):
            continue
        for c in p["spec"].get("containers", []):
            req = c.get("resources", {}).get("requests", {}) or {}
            lim = c.get("resources", {}).get("limits", {}) or {}
            n += int(req.get("nvidia.com/gpu") or lim.get("nvidia.com/gpu") or 0)
    return n


def logs(site: Site, job: str) -> str:
    return kubectl(site, "logs", f"job/{job}").stdout
