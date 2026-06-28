"""
Pipeline parameters — defaults match the hardcoded values in src/preprocessing.py
and src/features.py exactly so TracedPipeline(default_params) reproduces them.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class PipelineParams:
    # ── CLAHE ─────────────────────────────────────────────────────────────
    clahe_clip: float = 3.0          # clipLimit
    clahe_tile: int   = 8            # tileGridSize (square)

    # ── Skin segmentation (YCrCb morphology) ─────────────────────────────
    skin_close: int = 9              # MORPH_CLOSE kernel size (ellipse)
    skin_open:  int = 5              # MORPH_OPEN  kernel size (ellipse)

    # ── Adaptive lesion threshold multipliers ─────────────────────────────
    cr_k:    float = 1.5             # thr_cr = max(cr_floor, mu_cr + cr_k * std_cr)
    a_k:     float = 1.2             # thr_a  = max(a_floor,  mu_a  + a_k  * std_a)
    cr_floor: float = 145.0          # floor for Cr threshold
    a_floor:  float = 133.0          # floor for a* threshold

    # ── Red mask morphology ───────────────────────────────────────────────
    red_open:  int = 3               # MORPH_OPEN  kernel size
    red_close: int = 7               # MORPH_CLOSE kernel size

    # ── Lesion shape / size filters ───────────────────────────────────────
    max_aspect:        float = 3.5   # max width/height ratio
    min_fill:          float = 0.25  # min blob-area / bbox-area
    max_area_pct:      float = 0.04  # max blob area as fraction of skin pixels
    min_local_contrast: float = 1.5  # min a* difference vs surrounding ring

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineParams":
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


DEFAULT_PARAMS = PipelineParams()
