"""Unit tests for the pure parts: sizing, preflight, volume manifests, podscript, recorder parsing.
The policy values here are arbitrary test numbers, not any cluster's policy."""

import base64
import subprocess
import sys

import pytest

from nrp_kit import podscript, preflight, sizing, usage_recorder
from nrp_kit.config import Policy, Site
from nrp_kit.volumes import VolumeSet

POL = Policy(cpu_band=(0.25, 2.0), memory_band=(0.25, 1.5), small_class_cpu=1, small_class_mem_gib=2,
             gpu_busy_floor=0.5, max_full_gpu_pods=4)
SITE = Site(namespace="test-ns", policy=POL)


def test_no_measurement_declines():
    assert isinstance(sizing.request_for(POL, None, 2), sizing.Decline)


def test_small_class():
    r = sizing.request_for(POL, sizing.Usage(0.4, 0.8, 1.2), 1)
    assert isinstance(r, sizing.Request) and r.small_class and r.cpu == 1 and r.memory_gib <= 2


def test_measured_request_is_fair():
    u = sizing.Usage(1.0, 4.0, 4.4)
    r = sizing.request_for(POL, u, 2)
    assert isinstance(r, sizing.Request) and not r.small_class
    assert r.memory_gib >= 4.4 and sizing.check(POL, r.cpu, r.memory_gib, u) == []


def test_bursty_declines():
    assert isinstance(sizing.request_for(POL, sizing.Usage(1.0, 1.0, 10.0), 2), sizing.Decline)


def test_check_flags_overrequest():
    assert sizing.check(POL, 8, 16, sizing.Usage(0.5, 1.0, 1.0))


class R:
    def __init__(self, cpu="1", memory="2Gi", gpu=0, eph="4Gi"):
        self.cpu, self.memory, self.gpu, self.ephemeral_storage = cpu, memory, gpu, eph


class V:
    def __init__(self, claim):
        self.claim_name, self.mount_path, self.name = claim, "/w0", "v0"


class D:
    def __init__(self, cmd, res, vols=(), sel=None, backoff=0):
        self.name, self.command, self.args, self.resources = "j", cmd, [], res
        self.volumes, self.node_selector, self.backoff_limit, self.labels = list(vols), sel or {}, backoff, {}


GOOD = ["/bin/bash", "-c", "timeout 60 python3 x.py"]


def test_preflight_good_cpu():
    assert preflight.check(SITE, D(GOOD, R())) == []


def test_preflight_sleep_and_timeout():
    bad = preflight.check(SITE, D(["/bin/bash", "-c", "python3 x.py; sleep 10"], R()))
    assert any("sleeps" in b for b in bad) and any("timeout" in b for b in bad)


def test_preflight_gpu_download_and_unmeasured():
    d = D(["/bin/bash", "-c", "timeout 60 pip install x"], R(cpu="2", memory="5Gi", gpu=1))
    bad = preflight.check(SITE, d, gpus_in_use=0)
    assert any("installs or downloads" in b for b in bad) and any("neither measured" in b for b in bad)


def test_preflight_gpu_pilot_ok_and_cap():
    d = D(GOOD, R(cpu="2", memory="5Gi", gpu=1))
    assert preflight.check(SITE, d, pilot_reason="stated", gpus_in_use=0) == []
    assert any("exceed" in b for b in preflight.check(SITE, d, pilot_reason="s", gpus_in_use=4))
    assert any("not counted" in b for b in preflight.check(SITE, d, pilot_reason="s"))


def test_preflight_volume_zone():
    d = D(GOOD, R(), vols=[V("c")])
    assert any("zone" in b for b in preflight.check(SITE, d))
    assert preflight.check(SITE, D(GOOD, R(), vols=[V("c")], sel={"topology.kubernetes.io/zone": "z"})) == []


def test_preflight_cpu_above_small_class_needs_measurement():
    d = D(GOOD, R(cpu="2", memory="5Gi"))
    assert preflight.check(SITE, d) and preflight.check(SITE, d, sized_by="measured") == []


def test_volume_set():
    vs = VolumeSet("t-w0", 4, 8, 5, "block-class", "z", "app1")
    assert vs.names() == ["t-w0-0", "t-w0-1", "t-w0-2", "t-w0-3"]
    assert vs.shard_dirs() == "/w0/shards,/w1/shards,/w2/shards,/w3/shards"
    m = vs.manifest(SITE)
    assert m.count("kind: PersistentVolumeClaim") == 4 and "storage: 8Gi" in m and "namespace: test-ns" in m


def test_podscript_roundtrip(tmp_path):
    text = "print('a \\\\ b \"q\"')\n"
    cmd = podscript.script(podscript.embed_file(text, str(tmp_path / "x.py")))
    assert cmd[0] == "/bin/bash" and cmd[2].startswith("set -euo pipefail")
    b = cmd[2].split("echo ")[1].split(" |")[0]
    assert base64.b64decode(b).decode() == text


def test_embed_module_stage_model():
    s = podscript.embed_module("stage_model.py", "/tmp/s.py")
    assert s.startswith("echo ") and "base64 -d" in s


def test_recorder_parsing_and_assess():
    assert usage_recorder.parse_cpu("250m") == pytest.approx(0.25)
    assert usage_recorder.parse_mem("2Gi") == 2 * 2**30
    h = [(0.1, 2**30)] * 10
    m, low = usage_recorder.assess(h, 2000, 2, 8 * 2**30, grace=900, window=10, floor=0.25,
                                   small_cpu=1, small_mem=2 * 2**30)
    assert m and low
    m, low = usage_recorder.assess(h, 100, 2, 8 * 2**30, grace=900, window=10, floor=0.25,
                                   small_cpu=1, small_mem=2 * 2**30)
    assert m and not low


def test_stage_model_compiles():
    import importlib.resources
    src = importlib.resources.files("nrp_kit").joinpath("stage_model.py").read_text("utf-8")
    compile(src, "stage_model.py", "exec")
