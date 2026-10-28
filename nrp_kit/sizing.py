"""Size a pod's requests from what a pod of its class measurably used, or decline to.

A shared cluster asks that a pod use a fair share of what it requests: mean use inside a band
of the request, for CPU and for memory. With requests equal to limits, a pod also runs out of
memory at its memory request, so the request has to cover the peak while the mean stays inside
the band. That gives an explicit window for memory:

    request >= peak * headroom          (or the pod runs out of memory)
    request <= mean / band_low          (or the pod holds more than it uses)

so a class can be sized only when its peak is close enough to its mean. CPU has one side, since
a pod cannot use more than a request equal to its limit:

    request <= mean_cores / band_low

Every function here takes a measurement (see usage_recorder.py), never an estimate, and
``request_for`` returns a ``Decline`` with the reason when there is no measurement or no
request fits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Policy

MARGIN = 1.25  # aim this far inside the band, so one slow phase stays inside it
PEAK_HEADROOM = 1.10  # the memory request covers the measured peak by this much


@dataclass(frozen=True)
class Usage:
    """What a pod of this class did, averaged over its whole life, startup included."""

    mean_cpu_cores: float
    mean_mem_gib: float
    peak_mem_gib: float

    @property
    def burstiness(self) -> float:
        return self.peak_mem_gib / max(self.mean_mem_gib, 1e-9)


@dataclass(frozen=True)
class Request:
    cpu: int
    memory_gib: int
    small_class: bool
    note: str

    def __str__(self):
        tag = " (small class)" if self.small_class else ""
        return f"cpu={self.cpu} memory={self.memory_gib}Gi{tag}: {self.note}"


@dataclass(frozen=True)
class Decline:
    reason: str

    def __str__(self):
        return f"DECLINED: {self.reason}"


def cpu_request(policy: Policy, mean_cpu_cores: float, want: int) -> int:
    """The largest request up to ``want`` that the measured mean keeps inside the band."""
    return max(1, min(want, int(mean_cpu_cores / (policy.cpu_band[0] * MARGIN))))


def memory_window(policy: Policy, usage: Usage) -> tuple[float, float]:
    """(smallest safe, largest fair) memory request in GiB."""
    return usage.peak_mem_gib * PEAK_HEADROOM, usage.mean_mem_gib / (policy.memory_band[0] * MARGIN)


def request_for(policy: Policy, usage: Usage | None, want_cpu: int) -> Request | Decline:
    """A fair request for a class with this measured usage, or the reason there is none."""
    if usage is None:
        return Decline("no measured usage for this class; run one pod of it first and record it")
    if usage.mean_cpu_cores <= 0 or usage.mean_mem_gib <= 0:
        return Decline(f"the measurement is empty ({usage})")
    lo, hi = memory_window(policy, usage)
    if usage.peak_mem_gib * PEAK_HEADROOM <= policy.small_class_mem_gib and (
        want_cpu <= policy.small_class_cpu or usage.mean_cpu_cores < 1
    ):
        mem = min(policy.small_class_mem_gib, max(math.ceil(lo), 1))
        return Request(int(policy.small_class_cpu), int(mem), True, "fits the small class")
    if lo > hi:
        return Decline(
            f"peak {usage.peak_mem_gib:.1f} GiB is {usage.burstiness:.1f}x the mean "
            f"{usage.mean_mem_gib:.1f} GiB, more than one request can cover fairly. Make the peak "
            "shorter (stream instead of holding) or the steady state fuller."
        )
    mem = math.ceil(lo)
    if mem > hi:
        mem = math.floor(hi)
    cpu = cpu_request(policy, usage.mean_cpu_cores, want_cpu)
    left = check(policy, cpu, mem, usage)
    if left:
        return Decline(
            f"even cpu={cpu}, memory={mem}Gi would leave: " + "; ".join(left)
            + ". Fit the class into the small class, or shorten its waiting phases."
        )
    note = (f"mean {usage.mean_cpu_cores:.2f} cores, mean {usage.mean_mem_gib:.1f} / "
            f"peak {usage.peak_mem_gib:.1f} GiB")
    if cpu < want_cpu:
        note += f"; {want_cpu} CPUs asked, the mean supports {cpu}"
    return Request(cpu, mem, False, note)


def check(policy: Policy, request_cpu: float, request_mem_gib: float, usage: Usage) -> list[str]:
    """How a pod with this request and this usage would sit outside the bands (empty if fair)."""
    out = []
    if request_cpu <= policy.small_class_cpu and request_mem_gib <= policy.small_class_mem_gib:
        return out
    lo_c, hi_c = policy.cpu_band
    lo_m, hi_m = policy.memory_band
    fc, fm = usage.mean_cpu_cores / request_cpu, usage.mean_mem_gib / request_mem_gib
    if fc < lo_c or fc > hi_c:
        out.append(f"cpu at {100 * fc:.0f}% of {request_cpu:g}")
    if fm < lo_m or fm > hi_m:
        out.append(f"memory at {100 * fm:.0f}% of {request_mem_gib:g}Gi")
    if usage.peak_mem_gib > request_mem_gib:
        out.append(f"peak memory {usage.peak_mem_gib:.1f} GiB above the {request_mem_gib:g}Gi request")
    return out
