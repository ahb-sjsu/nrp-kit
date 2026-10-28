"""Site settings, read from a local JSON file that is not part of this repository.

The file is named by the environment variable NRP_KIT_CONFIG, or defaults to
~/.config/nrp-kit.json. It holds the namespace, the NATS endpoint of the nats-bursting
controller, and the usage policy of the cluster as its operators publish it. Nothing in this
package hard-codes a cluster's policy: copy the values from your cluster's current usage policy
page into the file, and update them when the page changes. See config.example.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Policy:
    """The published usage policy, as fractions of what a pod requests."""

    cpu_band: tuple[float, float]  # mean CPU use / CPU request must sit in this band
    memory_band: tuple[float, float]  # mean memory use / memory request must sit in this band
    small_class_cpu: float  # pods at or below both of these are outside the bands
    small_class_mem_gib: float
    gpu_busy_floor: float  # a GPU pod's GPU should be at least this busy, on average
    max_full_gpu_pods: int  # GPU pods one namespace may run at full use at once


@dataclass(frozen=True)
class Site:
    namespace: str
    policy: Policy
    nats_url: str = "nats://localhost:4222"
    kubectl: str = "kubectl"
    request_timeout_s: int = 60
    labels: dict = field(default_factory=dict)  # added to every Job this package submits


def path() -> str:
    return os.environ.get("NRP_KIT_CONFIG", os.path.expanduser("~/.config/nrp-kit.json"))


def load(p: str | None = None) -> Site:
    p = p or path()
    try:
        d = json.load(open(p, encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"nrp-kit: no site config at {p} ({e}); see config.example.json") from None
    pol = d["policy"]
    return Site(
        namespace=d["namespace"],
        policy=Policy(
            cpu_band=tuple(pol["cpu_band"]),
            memory_band=tuple(pol["memory_band"]),
            small_class_cpu=float(pol["small_class_cpu"]),
            small_class_mem_gib=float(pol["small_class_mem_gib"]),
            gpu_busy_floor=float(pol["gpu_busy_floor"]),
            max_full_gpu_pods=int(pol["max_full_gpu_pods"]),
        ),
        nats_url=d.get("nats_url", "nats://localhost:4222"),
        kubectl=d.get("kubectl", "kubectl"),
        request_timeout_s=int(d.get("request_timeout_s", 60)),
        labels=dict(d.get("labels", {})),
    )
