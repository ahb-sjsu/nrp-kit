"""Check a Job descriptor before it is submitted, as code rather than as memory.

``check`` returns a list of problems; an empty list means the descriptor passes. ``submit.submit``
refuses any descriptor with problems. The checks:

- the Job ends by itself: its long steps are wrapped in ``timeout``, and it never sleeps;
- ephemeral storage is declared, and the Job does not retry silently (backoff 0);
- a GPU Job installs and downloads nothing, so its GPU starts work at once: environments,
  code and weights are staged beforehand by small CPU Jobs onto volumes;
- a CPU Job fits the small class unless its request came from a measurement (sizing.py);
- a GPU Job's request came from a measurement, or is explicitly the first pod of its class
  (a pilot, with the stated reasoning) while a usage recorder watches it;
- a Job mounting single-attach block volumes is pinned to the volumes' zone;
- the namespace stays within its number of full-use GPU pods.
"""

from __future__ import annotations

import re

from .config import Site

DOWNLOADS = ("pip install", "git clone", "apt-get", "curl ", "wget ", "urlopen", "snapshot_download",
             "hf_hub_download", "conda install")


def _gib(q: str) -> float:
    q = str(q)
    for suf, mul in (("Gi", 1), ("Mi", 1 / 1024), ("G", 1e9 / 2**30), ("M", 1e6 / 2**30)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mul
    return float(q) / 2**30


def check(site: Site, desc, *, sized_by: str | None = None, pilot_reason: str | None = None,
          zone_key: str = "topology.kubernetes.io/zone", gpus_in_use: int | None = None) -> list[str]:
    """Problems with ``desc`` (a nats_bursting.JobDescriptor).

    ``sized_by``: how the request was obtained (str(sizing.Request)); None if not measured.
    ``pilot_reason``: for the first GPU pod of a class only, the stated basis of its request.
    ``gpus_in_use``: GPU pods already Pending/Running in the namespace (kube.gpu_pods).
    """
    p, bad = site.policy, []
    r = desc.resources
    script = " ".join(list(desc.command) + list(desc.args))
    gpu = int(r.gpu or 0) > 0
    if not r.ephemeral_storage:
        bad.append("ephemeral storage is not declared")
    if desc.backoff_limit != 0:
        bad.append("backoff_limit is not 0 (retries belong in the driver, where they are visible)")
    if re.search(r"\bsleep\b", script):
        bad.append("the command sleeps; a Job must work and end")
    if "timeout " not in script:
        bad.append("no step is bounded by timeout")
    if gpu:
        hits = [w.strip() for w in DOWNLOADS if w in script]
        if hits:
            bad.append(f"a GPU Job installs or downloads ({', '.join(hits)}); stage it first")
        if not sized_by and not pilot_reason:
            bad.append("a GPU Job's request is neither measured nor a stated pilot")
        if gpus_in_use is None:
            bad.append("GPU pods in the namespace were not counted")
        elif gpus_in_use + int(r.gpu) > p.max_full_gpu_pods:
            bad.append(f"{gpus_in_use} GPU pods already in the namespace; this would exceed "
                       f"{p.max_full_gpu_pods}")
    else:
        small = float(r.cpu) <= p.small_class_cpu and _gib(r.memory) <= p.small_class_mem_gib
        if not small and not sized_by:
            bad.append("a CPU Job above the small class without a measured request")
    if desc.volumes and any(v.claim_name for v in desc.volumes) and zone_key not in desc.node_selector:
        bad.append("mounts volumes but is not pinned to their zone")
    return bad
