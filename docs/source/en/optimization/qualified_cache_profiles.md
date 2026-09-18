# Fork-qualified cache profiles

This document is owned by this fork and records route-specific qualification evidence.
It intentionally does **not** duplicate upstream Diffusers' generic cache documentation.

For general MagCache semantics, calibration, and API usage, use the upstream Diffusers
caching guide. This file only records fork-qualified model/checkpoint/scheduler/runtime
profiles and the exact assumptions under which they were measured.

When this fork is rebased or rewritten against a newer upstream Diffusers version:

1. take generic cache infrastructure and documentation from upstream first;
2. drop local compatibility seams that upstream now provides;
3. reapply only the still-relevant fork-owned profile evidence;
4. requalify any profile whose model, scheduler, CFG topology, precision/quantization,
   cache implementation, or step semantics changed.

## HunyuanVideo 1.5 MagCache research profile

This fork registers `HunyuanVideo15TransformerBlock` in the generic transformer-block metadata registry so native block-cache methods can use the model's existing `CacheMixin` path. The project separately qualified native MagCache on the pinned `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` revision `286be7ce72277246578a3e3cc2487e95ddae5bcf` with 50 denoising steps and sequential classifier-free guidance.

The exact qualified 50-step ratio curve was the arithmetic mean of the conditional and unconditional calibration arrays from the project calibration run:

```python
HUNYUANVIDEO15_MAG_RATIOS_50 = [
    1.0, 0.9931376874446869, 1.0089889168739319, 1.0011072158813477,
    0.9982506036758423, 0.9972451627254486, 0.9977455735206604, 0.9984566569328308,
    0.9984758496284485, 0.9973903000354767, 0.9952424168586731, 0.9970934092998505,
    0.9978866577148438, 0.997559666633606, 0.9954802691936493, 0.9980159997940063,
    0.9978238940238953, 0.9976930916309357, 0.9967295229434967, 0.9959645569324493,
    0.9964526295661926, 0.9960361123085022, 0.9961098730564117, 0.9956640303134918,
    0.9950298964977264, 0.9955315589904785, 0.9946348965167999, 0.9955540299415588,
    0.9935533404350281, 0.9928642511367798, 0.9942461550235748, 0.9937227666378021,
    0.9931625127792358, 0.99092236161232, 0.9912545680999756, 0.9895545244216919,
    0.9898844659328461, 0.988347202539444, 0.9888069331645966, 0.9856226742267609,
    0.9837322235107422, 0.9824825823307037, 0.9774312078952789, 0.9778275787830353,
    0.9737721383571625, 0.9693790674209595, 0.9616954922676086, 0.9507126212120056,
    0.9443752765655518, 0.9402337074279785,
]
```

Two research profiles are retained:

```python
mag_fast = MagCacheConfig(
    threshold=0.04225,
    max_skip_steps=1,
    retention_ratio=0.40,
    num_inference_steps=50,
    mag_ratios=HUNYUANVIDEO15_MAG_RATIOS_50,
)

mag_balanced = MagCacheConfig(
    threshold=0.02175,
    max_skip_steps=1,
    retention_ratio=0.50,
    num_inference_steps=50,
    mag_ratios=HUNYUANVIDEO15_MAG_RATIOS_50,
)
```

On the project L4/NF4 research representation at 480x848 / 17 frames, `mag_fast` measured about 1.38x end-to-end speedup with latent cosine 0.999255 / 0.999449 on the two prompt-transfer cases. `mag_balanced` measured about 1.21-1.22x with cosine 0.999605 / 0.999727. Decoded Prompt-A pixel cosine was 0.999754 for `mag_fast` and 0.999890 for `mag_balanced`.

Long-temporal checks retained 121 frames and 50 steps while reducing only spatial cost. At 128x224, `mag_fast` measured about 1.383x with cosine 0.996888 and `mag_balanced` about 1.218x with cosine 0.997038. A one-load spatial DOE through 224x384 kept both profiles stable without pruning.

> [!WARNING]
> These are fork-specific research profiles, not portable defaults. They were qualified with the fixed ratio curve above, the pinned HunyuanVideo 1.5 checkpoint, sequential CFG, the project NF4/BF16 representation, and the tested scheduler/step count. The generic sequential-CFG calibration guidance above recommends the first conditional array in most cases; changing this Hunyuan profile's qualified ratio policy, checkpoint, scheduler, guidance topology, precision/quantization, attention backend, adapters, or denoising-step count requires requalification. A full 480x848 / 121-frame Cartesian confirmation was not required for this research closure.
