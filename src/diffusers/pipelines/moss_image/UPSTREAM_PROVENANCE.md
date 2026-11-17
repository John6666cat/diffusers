# MOSS-Image pipeline provenance

The initial `pipeline_moss_image.py` is derived from the MOSS-Image-Base upstream source snapshot:

- Repository: `OpenMOSS-Team/MOSS-Image-Base`
- Snapshot Git commit: `85975793d8b3d432f1b6e27f352abaa4c2a4c0d3`
- Upstream file: `pipeline_moss_image.py` (also duplicated under model component subfolders)
- Upstream license: Apache-2.0, preserved in `LICENSE_OPENMOSS.txt`

The local changes add `CacheMixin` and pipeline-owned per-lane `cache_context` around the existing denoising calls. Cache OFF retains the official compute path. `diffusers.hooks.spectrum_moss` is a separate research-only adapter using the local Diffusers 0.43.0 Spectrum predictor and does not reproduce GPL-licensed ComfyUI implementation code.

This is **not** an official Diffusers pipeline release, an upstream model update, a product deployment, or a verified GPU/quality profile. The original model's Hub `model_index.json` still names the original custom pipeline. For this staged fork, import these classes explicitly and load each official component through the documented model subfolders; do not silently substitute the upstream custom pipeline module.
