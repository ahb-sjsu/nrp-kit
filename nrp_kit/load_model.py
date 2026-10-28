"""Load a Hugging Face causal LM from a volume set straight onto the GPU, in a small pod.

Runs inside a GPU pod (copy it in with ``podscript.embed_module("load_model.py", dest)``). The
model is built with no memory (meta device) and allocated empty on the GPU. The weight files are
then read in parallel, one thread per file (one file per volume, as ``stage_model`` lays them
out), without memory-mapping: each tensor is read in 64 MB pieces into one pinned buffer per
thread and copied into its place, and the page cache behind it is dropped. Host memory therefore
stays at a few pieces however large the model is, so the pod fits the cluster's small class
(1 CPU, 2 GiB) and the GPU starts working within seconds of the pod starting. Buffers that are
computed rather than stored (rotary frequencies) are rebuilt on the device.

    from load_model import load_model
    model = load_model("/w0/model", sorted(glob.glob("/w?/shards/*.safetensors")), "cuda")

    python load_model.py --model /w0/model --shards /w?/shards/*.safetensors   # load and time it

Measured on an A10 with a 7B bf16 model on four linstor volumes: loaded 20 s after container
start, host peak 1.55 GiB. Supports bf16/f16/f32 safetensors whose dtype matches the model's.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
CHUNK = 64 << 20


def _read_file(path: str, targets: dict, pinned: bool) -> list[str]:
    """Copy every tensor in one safetensors file into its parameter. Returns the keys read."""
    keys = []
    buf = torch.empty(CHUNK, dtype=torch.uint8, pin_memory=pinned)
    view = memoryview(buf.numpy())
    with open(path, "rb", buffering=0) as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        base = 8 + n
        for k, m in header.items():
            if k == "__metadata__":
                continue
            if k not in targets:
                raise ValueError(f"{path}: {k} is not a parameter of the model")
            t = targets[k]
            if DTYPES.get(m["dtype"]) != t.dtype or list(m["shape"]) != list(t.shape):
                raise ValueError(f"{k}: file {m['dtype']} {m['shape']} vs model {t.dtype} {list(t.shape)}")
            start, stop = m["data_offsets"]
            total = stop - start
            dst = t.data.reshape(-1).view(torch.uint8)
            f.seek(base + start)
            pos = 0
            while pos < total:
                r = min(CHUNK, total - pos)
                if f.readinto(view[:r]) != r:
                    raise ValueError(f"{path}: short read in {k}")
                dst[pos:pos + r].copy_(buf[:r])
                pos += r
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(f.fileno(), base + start, total, os.POSIX_FADV_DONTNEED)
            keys.append(k)
    return keys


def _rebuild_meta_buffers(model, cfg, device) -> None:
    """Replace each module that still holds buffers on meta (they are computed at construction,
    not stored in the files) with a fresh one built on ``device`` from the config."""
    for name, mod in list(model.named_modules()):
        if not any(b.device.type == "meta" for b in mod.buffers(recurse=False)):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        try:
            setattr(parent, attr, type(mod)(config=cfg, device=device))
        except TypeError as e:
            raise ValueError(f"cannot rebuild {name} ({type(mod).__name__}) on {device}: {e}") from e
    left = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if left:
        raise ValueError(f"buffers still on meta: {left}")


def load_model(model_dir: str, shards: list[str], device: str = "cuda", dtype=torch.bfloat16):
    """The model in ``model_dir`` (config files) with weights from ``shards``, on ``device``,
    in eval mode. Every parameter must come from exactly the files given."""
    cfg = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    model.to_empty(device=device)
    model.tie_weights()  # moving off meta can untie shared weights; tie them again
    targets = dict(model.named_parameters(remove_duplicate=False))
    pinned = torch.device(device).type == "cuda"
    with ThreadPoolExecutor(max(1, len(shards))) as ex:
        per_file = list(ex.map(lambda p: _read_file(p, targets, pinned), shards))
    loaded = [k for ks in per_file for k in ks]
    if len(loaded) != len(set(loaded)):
        raise ValueError("a weight appears in more than one file")
    have = {id(targets[k]) for k in loaded}
    missing = {k for k, p in model.named_parameters() if id(p) not in have}
    if missing:
        raise ValueError(f"weights missing from the files: {sorted(missing)[:5]}")
    _rebuild_meta_buffers(model, cfg, device)
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Load a model from a volume set and report the time.")
    ap.add_argument("--model", required=True, help="directory with the config and tokenizer files")
    ap.add_argument("--shards", nargs="+", required=True, help="the weight files, one per volume")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t = time.time()
    model = load_model(a.model, a.shards, a.device)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    n = sum(p.numel() for p in model.parameters())
    print(f"LOAD_OK params={n} seconds={time.time() - t:.1f}", flush=True)


if __name__ == "__main__":
    main()
