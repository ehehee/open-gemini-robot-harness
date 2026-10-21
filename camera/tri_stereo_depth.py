"""TRI Stereo learned depth (from third_party/raiden) as an alternative to the
ZED SDK's own depth for the overhead camera -- addresses the noisy/curved
table surface seen with classical stereo (ULTRA): see
third_party/raiden/docs/guide/tri_stereo.md.

Loaded by file path rather than `pip install`-ing third_party/raiden: that
package's pyproject.toml requires Python >=3.11,<3.12, which doesn't exist on
this machine, but raiden/depth/tri_stereo.py has zero internal `raiden.*`
imports (only cv2/numpy/torch/onnxruntime), so it works standalone without
the rest of the package or its install constraints.

Needs onnxruntime-gpu + torch in THIS project's venv (see requirements.txt) --
torch is imported first by tri_stereo.py itself so its bundled cuDNN is loaded
before onnxruntime looks for libcudnn.so.9 via dlopen, same trick noted in
tri_stereo.py's own _ensure_loaded().

pyzed itself still needs numpy<2 (hard ABI constraint on its compiled
extension) while i2rt declares numpy==2.2.6; this project's venv runs numpy
1.26.4 and i2rt works fine on it in practice -- see yam_control/yam_backend.py
and this repo's own verification of that combination. Don't "fix" this by
bumping numpy back to 2.x; that breaks pyzed with a binary-incompatibility
crash, not just a warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_RAIDEN_DIR = Path(__file__).resolve().parents[1] / "third_party" / "raiden"


def _load_predictor_class():
    if str(_RAIDEN_DIR) not in sys.path:
        sys.path.insert(0, str(_RAIDEN_DIR))
    from raiden.depth.tri_stereo import TRIStereoOnnxDepthPredictor

    return TRIStereoOnnxDepthPredictor


class TriStereoDepth:
    """variant: 'c64' (higher quality, default) or 'c32' (faster, lighter)."""

    def __init__(self, variant: str = "c64"):
        predictor_cls = _load_predictor_class()
        if not predictor_cls.model_available(variant):
            raise RuntimeError(
                f"TRI Stereo {variant} ONNX model not found under {_RAIDEN_DIR}/weights/tri_stereo/. "
                "Run `git lfs pull` in that checkout."
            )
        self._predictor = predictor_cls(variant=variant)

    def predict(self, left_bgr: np.ndarray, right_bgr: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
        """Returns a float32 (H, W) depth map in meters, 0 = invalid, at the
        ORIGINAL left-image resolution (the predictor handles internal
        resize-to-model-input-and-back itself)."""
        return self._predictor.predict(left_bgr, right_bgr, fx, baseline_m)
