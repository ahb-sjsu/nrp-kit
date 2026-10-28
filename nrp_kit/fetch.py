"""Bring a directory from a volume back to the submitting host, through small CPU Jobs' logs.

Each part is one small Job that tars the directory (once, under a name holding its sha256),
prints one chunk as base64 between markers, and ends. The host reassembles the chunks, checks
the sha256 and unpacks. Chunks stay well under typical log-rotation sizes.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import subprocess

from . import kube, podscript, submit

CHUNK = 6 << 20  # raw bytes per part: 8 MiB of base64


def part_script(src: str, tag: str, part: int) -> list[str]:
    return podscript.script(f"""
cd {src}/..
if [ {part} -eq 0 ] || ! ls .fetch/{tag}.*.tgz >/dev/null 2>&1; then
  mkdir -p .fetch
  timeout 600 tar -czf .fetch/{tag}.tmp {os.path.basename(src)}
  h=$(sha256sum .fetch/{tag}.tmp | cut -c1-64)
  rm -f .fetch/{tag}.*.tgz && mv .fetch/{tag}.tmp .fetch/{tag}.$h.tgz
fi
f=$(ls .fetch/{tag}.*.tgz)
echo "FETCH_FILE $(basename $f) $(stat -c %s $f)"
echo FETCH_BEGIN
timeout 600 dd if=$f bs={CHUNK} skip={part} count=1 status=none | base64 -w0
echo
echo FETCH_END
""")


def fetch(site, make_descriptor, src: str, tag: str, out_dir: str) -> str:
    """``make_descriptor(name, command)`` returns a small-class CPU JobDescriptor that mounts the
    volume holding ``src``. Returns the unpacked directory."""
    os.makedirs(out_dir, exist_ok=True)
    blob, part, total, fname = bytearray(), 0, None, None
    while total is None or len(blob) < total:
        name = f"{tag}-fetch-{part}"
        submit.submit(site, make_descriptor(name, part_script(src, tag, part)))
        submit.wait_finished(site, name)
        log = kube.logs(site, name)
        m = re.search(r"^FETCH_FILE (\S+) (\d+)$", log, re.M)
        body = re.search(r"FETCH_BEGIN\n(.*)\nFETCH_END", log, re.S)
        if not m or not body:
            raise SystemExit(f"{name}: no chunk in the log")
        if fname not in (None, m.group(1)):
            raise SystemExit(f"{name}: the archive changed between parts")
        fname, total = m.group(1), int(m.group(2))
        chunk = base64.b64decode(body.group(1))
        if len(chunk) != min(CHUNK, total - part * CHUNK):
            raise SystemExit(f"{name}: chunk of {len(chunk)} bytes; log truncated?")
        blob += chunk
        kube.kubectl(site, "delete", "job", name)  # finished and read
        part += 1
    want, got = fname.split(".")[1], hashlib.sha256(blob).hexdigest()
    if got != want:
        raise SystemExit(f"reassembled sha256 {got} != {want}")
    path = os.path.join(out_dir, fname)
    open(path, "wb").write(blob)
    subprocess.run(["tar", "-xzf", path, "-C", out_dir], check=True)
    return out_dir
