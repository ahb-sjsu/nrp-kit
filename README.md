# nrp-kit

Batch work on the [National Research Platform](https://nrp.ai) (Nautilus), done as a good
citizen of a shared cluster: every pod's request comes from what a pod of its class measurably
used, GPU pods start computing as soon as they start, every Job works and then ends, and
submissions go through the [nats-bursting](https://github.com/ahb-sjsu/nats-bursting)
controller, which applies polite backoff when the cluster is busy.

The package collects the pieces several of our projects had each written for themselves.

| module | what it does |
|---|---|
| `config` | Site settings from a local JSON file (namespace, NATS endpoint, and the cluster's published usage policy). Nothing cluster-specific is hard-coded. |
| `sizing` | Turns a measured class usage (mean CPU, mean and peak memory) into a fair request, or declines with the reason. |
| `usage_recorder` | Runs beside a batch on the submitting host. Records each Job's measured usage for `sizing`, and can stop our own pods that hold much more than they use. |
| `preflight` | Checks a Job descriptor before submission: ends by itself, bounded steps, no installs or downloads on GPU pods, measured or stated requests, zone pinning for block volumes, the namespace's GPU pod count. |
| `submit` | Submits through nats-bursting; never replaces an active Job; removes a finished Job of the same name first and checks the new Job is fresh. |
| `capacity` | Estimates where GPUs are likely free, per zone and model, from the node list and the cluster's public GPU metrics, without submitting anything. |
| `volumes` | Sets of small single-attach block volumes, one weight file per volume, read in parallel so a GPU pod's weights arrive quickly. |
| `stage_model` | Runs in a small CPU pod: stages a Hugging Face model onto a volume set and verifies every file's hash. |
| `load_model` | Runs in a GPU pod: builds a Hugging Face causal LM with no memory, then reads the volume set's weight files in parallel straight into the GPU, without memory-mapping, so a 7B model loads in a small-class pod (1 CPU, 2 GiB). Checked against `from_pretrained` in CI. |
| `podscript` | Builds Job commands with helpers embedded as base64, so nothing depends on shell quoting. |
| `fetch` | Brings a result directory back from a volume through small Jobs' logs, checked by sha256. |

## The workflow it supports

1. `python -m nrp_kit.capacity`: pick a zone and GPU model with likely free GPUs.
2. `VolumeSet(...).apply(site)`: create a volume set in that zone.
3. A small CPU Job runs `stage_model.py` onto the set: environment, code and weights, verified.
4. A pilot GPU Job, with a stated request (small class when it loads with `load_model`), runs a first slice while `usage_recorder` measures it.
5. `sizing.request_for(...)` sizes the remaining Jobs from the pilot's measurement.
6. The remaining slices run, at most the namespace's number of full-use GPU pods at a time.
7. `fetch` brings the outputs home; `VolumeSet.release` removes the volumes when no Job is active.

## Configuration

Copy `config.example.json` to `~/.config/nrp-kit.json` (or point `NRP_KIT_CONFIG` at it) and fill
in the policy section from your cluster's current published usage policy. The values change from
time to time; the policy page is the authority, not this file.

## Install

```bash
pip install -e .            # the pure modules
pip install -e .[submit]    # plus the nats-bursting client, for submit and fetch
pip install -e .[test] && pytest -q
```

## License

Two licenses, split by what the file is.

| what | license | file |
|---|---|---|
| Source code | MIT | [LICENSE](LICENSE) |
| Prose (README, other Markdown) and `config.example.json` | CC BY 4.0 | [LICENSE-TEXT](LICENSE-TEXT) |
