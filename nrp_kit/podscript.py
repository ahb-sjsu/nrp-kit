"""Build the bash a Job runs, with Python helpers embedded as base64 (no quoting to get wrong)."""

from __future__ import annotations

import base64
import importlib.resources
import shlex


def embed_file(text: str, dest: str) -> str:
    """Bash that writes ``text`` to ``dest`` inside the pod."""
    b = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"echo {b} | base64 -d > {shlex.quote(dest)}\n"


def embed_module(name: str, dest: str) -> str:
    """Bash that writes nrp_kit's module ``name`` (e.g. 'stage_model.py') to ``dest``."""
    return embed_file(importlib.resources.files("nrp_kit").joinpath(name).read_text("utf-8"), dest)


def script(*steps: str) -> list[str]:
    """A Job command: strict bash running ``steps`` in order."""
    return ["/bin/bash", "-c", "set -euo pipefail\n" + "\n".join(s.rstrip("\n") for s in steps) + "\n"]
