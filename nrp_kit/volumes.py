"""Sets of small single-attach block volumes, read in parallel by one pod.

A GPU pod should start computing soon after it starts, so its weights have to arrive fast. A
shared filesystem read by many pods is slow for one large file; several small block volumes,
each holding one weight file and read at the same time, are much faster. A VolumeSet is that
layout: volume k (mounted at /wk) holds weight file k+1 of n, and volume 0 also holds the small
files, the environment, code, inputs and outputs. Volumes attach to one pod at a time, so each
concurrent GPU pod needs
its own set, and every pod using a set is pinned to the volumes' zone.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import kube
from .config import Site


@dataclass(frozen=True)
class VolumeSet:
    prefix: str  # e.g. "sd-w0": volumes sd-w0-0 .. sd-w0-{n-1}
    n: int
    gib0: int  # volume 0: weight file 1 + small files + env + code + inputs + outputs
    gibk: int  # volumes 1..n-1: one weight file each
    storage_class: str
    zone: str  # the zone label value the class provisions in
    app: str  # label shared by the set and the Jobs using it

    def names(self) -> list[str]:
        return [f"{self.prefix}-{k}" for k in range(self.n)]

    def mounts(self) -> list[tuple[str, str]]:
        """(claim, mount path) pairs: volume k at /wk."""
        return [(c, f"/w{k}") for k, c in enumerate(self.names())]

    def shard_dirs(self) -> str:
        """--shard-dests for stage_model.py: weight file k of n on volume k-1."""
        return ",".join(f"/w{k}/shards" for k in range(self.n))

    def manifest(self, site: Site) -> str:
        return "".join(
            f"""---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {{name: {c}, namespace: {site.namespace}, labels: {{app: {self.app}}}}}
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: {self.storage_class}
  resources: {{requests: {{storage: {self.gib0 if k == 0 else self.gibk}Gi}}}}
"""
            for k, c in enumerate(self.names())
        )

    def apply(self, site: Site) -> str:
        r = kube.kubectl(site, "apply", "-f", "-", stdin=self.manifest(site))
        if r.returncode:
            raise SystemExit(r.stderr)
        return r.stdout

    def release(self, site: Site) -> str:
        """Delete the set, only when no Job with the set's app label is active."""
        act = kube.active_jobs(site, f"app={self.app}")
        if act:
            raise SystemExit(f"Jobs still active: {act}")
        return kube.kubectl(site, "delete", "pvc", *self.names()).stdout
