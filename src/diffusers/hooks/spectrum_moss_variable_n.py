"""Opt-in variable-step SPECTRUM schedule for MOSS-Image.

N=50 uses the reference F28 placement. Other step counts use the
same deterministic nested-anchor rule; a mask is not a quality guarantee.
"""


_WARMUP = 8
_REAL_TAIL = 2
_MIN_STEPS = 18
_N50_BASE = frozenset((8, 10, 12, 14, 16, 19, 22, 25, 26, 28,
                       30, 32, 34, 36, 38, 40, 42, 44, 46, 47))
_N50_F28 = (8, 10, 12, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 30,
            31, 32, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 46, 47)


def _half_even_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator or
                          (2 * remainder == denominator and quotient % 2 == 1))


def _real_anchor_priority(candidates: set[int]) -> list[int]:
    remaining = set(candidates)
    low, high = min(remaining), max(remaining)
    kept = []
    for fraction in (1 / 3, 2 / 3):
        if remaining:
            desired = low + (high - low) * fraction
            chosen = min(remaining, key=lambda step: (abs(step - desired), step))
            kept.append(chosen)
            remaining.remove(chosen)
    while remaining:
        chosen = max(remaining, key=lambda step: (min(abs(step - a) for a in kept), -step))
        kept.append(chosen)
        remaining.remove(chosen)
    return kept


def moss_variable_n_forecast_indices(num_inference_steps: int) -> tuple[int, ...]:
    """Return the F28-analogue mask for actual scheduler length N.

    N below 18 must use cache OFF; it is not silently assigned a different
    density. The first 8 and final 2 steps remain real at supported N.
    """
    if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, int):
        raise TypeError("MOSS num_inference_steps must be an integer")
    if num_inference_steps < 1:
        raise ValueError("MOSS num_inference_steps must be positive")
    if num_inference_steps < _MIN_STEPS:
        return ()

    n = num_inference_steps
    eligible = tuple(range(_WARMUP, n - _REAL_TAIL))
    interior = len(eligible)
    target_forecasts = _half_even_ratio(7 * interior, 10)
    if not 0 < target_forecasts <= interior - 2:
        raise ValueError("MOSS schedule requires two interior real anchors")

    if n == 50:
        base = set(_N50_BASE)
    else:
        half = max(1, _half_even_ratio(interior, 2))
        base = {eligible[min(interior - 1, int((i + 0.5) * interior / half))]
                for i in range(half)}

    # Preserve nested-anchor priority and promote the first
    # (interior - forecasts) real anchors, including two separated anchors.
    real_priority = _real_anchor_priority(set(eligible) - base)
    required_real = interior - target_forecasts
    forecast_mask = tuple(sorted(set(eligible) - set(real_priority[:required_real])))
    if len(forecast_mask) != target_forecasts or not base.issubset(forecast_mask):
        raise RuntimeError("MOSS nested-anchor mask contract failed")
    if n == 50 and forecast_mask != _N50_F28:
        raise RuntimeError("MOSS N50 reference F28 mask drifted")
    return forecast_mask
