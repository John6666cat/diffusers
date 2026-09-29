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

## SDXL ordinary variable-step SPECTRUM profile

This fork qualifies an opt-in variable-step profile for ordinary Stable Diffusion XL text-to-image inference with the
Euler scheduler. Enable it on the SDXL UNet and pass the same inference-step count to the pipeline:

```python
from diffusers import SpectrumCacheConfig

num_inference_steps = 32
config = SpectrumCacheConfig.for_sdxl_variable(num_inference_steps=num_inference_steps)
pipe.unet.enable_cache(config)

image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
```

The factory retains the ordinary SDXL predictor settings (`degree=4`, `ridge_lambda=0.1`, `blend_w=0.60`) and uses:

- `coordinate_policy="runtime_index_normalized"`;
- `warmup_steps=6`;
- `tail_actual_steps=3`;
- `max_consecutive_forecast_steps=3`.

The max-three guard is compiled from the ordinary adaptive schedule. If a forecast run is longer than three steps, the
midpoint of the first overlong run is promoted to a real forward pass; this repeats until every forecast run is at most
three steps. The guard only adds real-compute positions.

The qualification used the historical SDXL Base 1.0 route with the default Euler scheduler at 1024x1024 and guidance
scale 5.0. The measured/identity step set was `16/20/24/28/32/36/40`. The 16- and 20-step schedules were identity
controls where the max-three guard did not change the decision mask.

On the fresh broad 24/28/32/36/40-step holdout, the guarded profile beat the matched unguarded comparison schedule in
68 of 75 cases, with a mean image-PSNR delta of `+0.799557 dB`. A targeted 28/32/40-step multi-person and interaction
hard-case holdout produced 37 wins and 8 losses over 45 matched cases, with a mean delta of `+0.242122 dB`; the worst
measured regression was `-0.345157 dB`.

`SpectrumCacheConfig.for_sdxl()` is unchanged and remains on its historical `legacy_fixed_max` coordinate policy.
`for_sdxl_variable()` is an explicit opt-in, not a new route default.

> [!WARNING]
> This qualification is specific to ordinary SDXL with the Euler scheduler. The same coordinate/guard policy was not
> established as universal on SD1.5 PNDM or Wan2.1, and it does not cover PAG, the conservative SDXL preset, arbitrary
> unmeasured step counts, or few-step/distilled routes. Revalidate when changing the checkpoint, scheduler, guidance
> topology, precision/quantization, attention backend, adapters, conditioning path, or denoising-step regime.

## FLUX.2 Klein Base 4B SPECTRUM profile

The retained FLUX.2 Klein Base 4B profile is qualified for
`black-forest-labs/FLUX.2-klein-base-4B` revision
`a3b4f4849157f664bdbc776fd7453c2783562f4d` on the standard 50-step
classifier-free-guidance text-to-image route.

The route is intentionally lane-asymmetric:

- `cond` remains full-compute for all 50 denoising steps;
- `uncond` uses the explicit forecast schedule below;
- the forecaster is first-order (`degree=1`) with no spectral/local blend (`blend_w=0.0`);
- the final six logical steps remain full-compute.

```python
cond_config = SpectrumCacheConfig(
    num_inference_steps=50,
    forecast_step_indices=(),
    warmup_steps=6,
    window_size=2.0,
    flex_window=0.0,
    degree=1,
    ridge_lambda=0.1,
    blend_w=0.0,
    history_limit=100,
    coordinate_max=50.0,
    tail_actual_steps=6,
)

uncond_config = SpectrumCacheConfig(
    num_inference_steps=50,
    forecast_step_indices=(24, 26, 28, 30, 32, 34, 36, 38, 40, 42),
    warmup_steps=6,
    window_size=2.0,
    flex_window=0.0,
    degree=1,
    ridge_lambda=0.1,
    blend_w=0.0,
    history_limit=100,
    coordinate_max=50.0,
    tail_actual_steps=6,
)
```

The profile was selected before the upstream synchronization and then requalified after the semantic rebuild on
upstream Diffusers commit `9f1246971270c84dcbe71233edb7a519596a5d02`.
The migrated candidate reproduced the pre-sync image-quality metrics exactly on both retained 1024x1024 prompts:

- prompt 1: cosine `0.9981166124`, PSNR `30.8182545 dB`;
- prompt 2: cosine `0.9992572665`, PSNR `42.3831923 dB`.

On the current L4 confirmation the aggregate measured wall-time speedup was about `1.078x`.
Timing is environment-sensitive; the retained qualification gate is finite output, cosine at least `0.995`,
PSNR at least `30 dB`, ten actual uncond prediction calls, no guard/fallback activation, and an exact full-compute
baseline canary after cache teardown.

The current upstream target is one commit newer than the initial migration base
`80c7ed262aeffbeb43ef13ae04baeb9b84515a69`; the additional upstream commit
`9f1246971270c84dcbe71233edb7a519596a5d02` contains split-device component-placement fixes including
`Flux2KleinPipeline`. The profile was requalified after retargeting to that commit.

> [!WARNING]
> This is a fork-qualified route-specific profile, not a portable default. Revalidate when changing the Klein
> checkpoint/revision, denoising step count, scheduler semantics, guidance topology, precision/quantization,
> accelerator/offload policy, attention backend, LoRA/adapters, image/inpaint conditioning, or cache-context topology.

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

## Wan2.1 T2V 1.3B variable-step SPECTRUM profile

This fork qualifies an opt-in variable-step SPECTRUM profile for the pinned
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` revision
`0fad780a534b6463e45facd96134c9f345acfa5b` on the text-only T2V route. Wan's
native denoising loop supplies separate `cache_context("cond")` and
`cache_context("uncond")` lanes with logical `step_index` and runtime step-count
provenance.

```python
from diffusers import SpectrumCacheConfig

num_inference_steps = 50
config = SpectrumCacheConfig.for_wan21_13b_variable(num_inference_steps)
pipe.transformer.enable_cache(config)
video = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=num_inference_steps,
).frames[0]
```

The factory is intentionally bounded to `40 <= num_inference_steps <= 60` and uses:

- `coordinate_policy="runtime_horizon_normalized"` (`2*i/N - 1`);
- protected early real steps `round(0.22*N)`;
- protected tail real steps `round(0.20*N)`;
- an explicit interior schedule that forecasts two of every three steps;
- `degree=4`, `ridge_lambda=0.1`, `blend_w=0.5`, `history_limit=100`;
- the dense predictor backend used by the qualification run.

At N=50 this reproduces the historical `mid19_keep10` forecast positions exactly:
`[11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38]`.

Qualification anchors covered N=40/45/50/55/60, three fixed prompts, 192x320,
81 frames, sequential CFG, the project NF4/FP16-compute runtime representation,
and fixed exact-source UMT5 conditioning. Across all 15 measured latent comparisons,
mean cosine was about `0.997870`, worst cosine `0.996628`, mean normalized RMSE
`0.06207`, and mean measured wall-time speedup `1.583x` (worst `1.520x`). The N=45/55
bounded confirmation also decoded full videos; its mean video PSNR was about
`36.17 dB`, with worst full-video PSNR `29.85 dB`. Start/end latent canaries remained exact,
and no SPECTRUM guard, coordinate, or prediction fallback latched.

`for_wan21_13b_variable()` is an explicit opt-in profile. It does not change the
historical generic SPECTRUM defaults or the separately qualified Wan MagCache profiles.

> [!WARNING]
> This qualification is route- and environment-specific. It does not establish Wan 14B,
> Wan2.2, I2V/VACE/image-conditioned routes, arbitrary step counts outside 40..60,
> alternate schedulers/guidance topology, LoRA/adapters, non-empty attention kwargs,
> different precision/quantization, or different conditioning as qualified. Revalidate
> those changes separately.

## Wan2.1 T2V 1.3B MagCache profiles

The project bounded-reopened native MagCache for the pinned `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` revision `0fad780a534b6463e45facd96134c9f345acfa5b`. Wan already exposes separate `cache_context("cond")` and `cache_context("uncond")` scopes, and the qualified MagCache run followed the sequential-CFG guidance above by using the first, conditional calibration array.

The 81-frame confirmation used this 50-step conditional ratio curve:

```python
WAN21_13B_MAG_RATIOS_50 = [
    1.0, 1.0139240026474, 0.971612274646759, 0.9779471158981323,
    0.9794172048568726, 0.9805305600166321, 0.9810647964477539, 0.9807896614074707,
    0.9818068146705627, 0.9816883206367493, 0.9814188480377197, 0.9810559749603271,
    0.9810541272163391, 0.9809654951095581, 0.9807681441307068, 0.9801260232925415,
    0.9799965620040894, 0.9798394441604614, 0.9791591167449951, 0.9793544411659241,
    0.9786234498023987, 0.9782900810241699, 0.9781445860862732, 0.977333128452301,
    0.9771683812141418, 0.9765759110450745, 0.9764811396598816, 0.9759577512741089,
    0.975703775882721, 0.9746671319007874, 0.9751186966896057, 0.974176824092865,
    0.9734554290771484, 0.9737706184387207, 0.9727852940559387, 0.9723516702651978,
    0.9719029068946838, 0.9718255996704102, 0.9710003733634949, 0.9708519577980042,
    0.970937192440033, 0.9704956412315369, 0.9707760214805603, 0.9708070755004883,
    0.9708809852600098, 0.9727936387062073, 0.9743280410766602, 0.9847269058227539,
    1.0319546461105347, 1.1225619316101074,
]
```

The balanced project profile is:

```python
mag_balanced = MagCacheConfig(
    threshold=0.12,
    max_skip_steps=4,
    retention_ratio=0.30,
    num_inference_steps=50,
    mag_ratios=WAN21_13B_MAG_RATIOS_50,
)
```

A faster source-style opt-in is:

```python
mag_source_style = MagCacheConfig(
    threshold=0.12,
    max_skip_steps=4,
    retention_ratio=0.20,
    num_inference_steps=50,
    mag_ratios=WAN21_13B_MAG_RATIOS_50,
)
```

In the final same-run 192x320 / 81-frame / 50-step L4/NF4 confirmation, the existing SPECTRUM `mid19_keep10` control measured 1.465x speedup with cosine 0.998827 and normalized RMSE 0.0484. `mag_balanced` measured 1.785x with cosine 0.999234 and normalized RMSE 0.0396, strictly dominating that control on the measured speed and latent-quality metrics. `mag_source_style` measured 2.017x with cosine 0.998258 and normalized RMSE 0.0592, retaining a faster Pareto point.

> [!WARNING]
> These measurements are environment- and route-specific. The profiles are qualified only for the pinned Wan2.1 1.3B text-only T2V route, 50-step sequential CFG, corrected zero-tail conditioning contract, and the tested NF4/FP16-compute representation. Recalibrate and revalidate for Wan 14B, Wan2.2, I2V/VACE or image conditioning, a different scheduler or step count, LoRA/adapters, attention kwargs, precision/quantization, or a changed guidance topology.

## Z-Image Base SPECTRUM + FirstBlockCache conservative profile

The project qualified a conservative nested-cache profile for the pinned
`Tongyi-MAI/Z-Image` revision `aa9e0836a9ca3bd891d531de8bdc682140edf325`
using the serialized private NF4 runtime
`John6666/_spectrum_test_model@0458754d0a978a92cccf83c1f23576cda693e307`.

This route uses 28 denoising steps, CFG 4.0, the project NF4/BF16 representation,
and the pinned Z-Image Base scheduler contract. The cache composition order is
FirstBlockCache inner, SPECTRUM outer.

The qualified SPECTRUM control is:

```python
zimage_base_aggressive8 = SpectrumCacheConfig(
    num_inference_steps=28,
    warmup_steps=9,
    window_size=2.0,
    flex_window=0.0,
    degree=3,
    ridge_lambda=0.1,
    blend_w=0.5,
    history_limit=28,
    coordinate_max=28.0,
    tail_actual_steps=4,
    forecast_step_indices=[9, 11, 13, 15, 17, 19, 21, 23],
)
```

The retained conservative inner-cache profile is:

```python
zimage_base_fbc_conservative = FirstBlockCacheConfig(
    threshold=0.07861335986428125,
)
```

Across the four-case broad qualification suite (wide spatial composition, square
bilingual typography, portrait/person, and landscape/interior), the fixed FBC
threshold skipped logical step 7 on every case. SPECTRUM aggressive8 averaged
1.347x denoiser speedup versus full compute. Adding the conservative inner FBC
profile averaged 1.409x versus full compute and 1.046x incremental speed versus
SPECTRUM alone, with an observed incremental range of 1.044x to 1.050x.

The conservative profile's mean latent cosine versus full compute was 0.98209.
Relative to the SPECTRUM control, the mean cosine delta was -0.00263 and the
worst observed delta was -0.00599. Mean decoded-image PSNR versus full compute
was 25.57 dB; the mean PSNR delta versus SPECTRUM was -0.72 dB and the worst
observed delta was -1.14 dB.

Forced-full FBC + SPECTRUM was latent-byte-exact to SPECTRUM alone in both the
initial composition screen and the later portrait broad-qualification check.

A more aggressive threshold of `0.09765634765725` is intentionally **not**
retained as a qualified profile. Although it averaged about 1.097x incremental
speed over SPECTRUM, its skip pattern changed across content types and the
landscape/interior case regressed by about 0.0279 latent cosine and 4.02 dB
decoded PSNR relative to the SPECTRUM control.

> [!WARNING]
> This is a fork-specific research profile, not a portable default. Requalify if
> the Z-Image checkpoint, scheduler, denoising-step count, CFG/guidance topology,
> precision or quantization, attention backend, adapters, SPECTRUM schedule, FBC
> implementation, or cache-context semantics change.
