"""Submit a Job through the nats-bursting controller, safely.

- A Job that is still active is never deleted or replaced.
- A finished Job of the same name is removed first, because a name in use makes the controller
  skip the new Job; after submitting, the Job's creationTimestamp is checked to be fresh, so a
  watcher never mistakes yesterday's Job for today's.
- Every descriptor passes preflight.check first.
"""

from __future__ import annotations

import calendar
import time

from . import kube, preflight
from .config import Site


def submit(site: Site, desc, **preflight_args) -> None:
    from nats_bursting import Client  # the controller's client library

    desc.labels = {**site.labels, **desc.labels}
    if int(desc.resources.gpu or 0) > 0 and "gpus_in_use" not in preflight_args:
        preflight_args["gpus_in_use"] = kube.gpu_pods(site)
    bad = preflight.check(site, desc, **preflight_args)
    if bad:
        raise SystemExit(f"PREFLIGHT {desc.name}: " + "; ".join(bad))
    state = kube.job_state(site, desc.name)
    if state == "active":
        raise SystemExit(f"{desc.name} is active; an active Job is never replaced")
    if state == "finished":
        kube.kubectl(site, "delete", "job", desc.name)
    t0 = time.time()
    with Client(nats_url=site.nats_url) as c:
        c.submit(desc)
    for _ in range(60):
        r = kube.kubectl(site, "get", "job", desc.name, "-o", "jsonpath={.metadata.creationTimestamp}")
        ts = r.stdout.strip()
        if r.returncode == 0 and ts:
            created = calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
            if created < t0 - 120:
                raise SystemExit(f"{desc.name}: stale creationTimestamp {ts}; the new Job was not created")
            print(desc.name, "created", ts, flush=True)
            return
        time.sleep(5)  # runs on the submitting host, never in a pod
    raise SystemExit(f"{desc.name}: no Job appeared")


def wait_finished(site: Site, name: str, poll_s: float = 15) -> str:
    """Block until the Job finishes; returns its final state ('finished' or 'absent')."""
    while True:
        s = kube.job_state(site, name)
        if s != "active":
            return s
        time.sleep(poll_s)  # on the submitting host
