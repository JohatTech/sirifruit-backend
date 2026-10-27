import numpy as np
import pytest

from app.ml.preprocess import HIST_DIM, INPUT_SIZE, InvalidImageError, decode_image, preprocess
from tests.conftest import make_pack


def test_preprocess_shapes_and_types():
    rgb, hist = preprocess(make_pack("fresa"))
    assert rgb.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert rgb.dtype == np.uint8
    assert hist.shape == (HIST_DIM,)
    assert hist.dtype == np.float32
    assert hist.sum() == pytest.approx(1.0, abs=1e-4)


def test_histograms_differ_between_flavors():
    _, h_fresa = preprocess(make_pack("fresa"))
    _, h_pina = preprocess(make_pack("pina"))
    _, h_fresa2 = preprocess(make_pack("fresa", seed=7))
    same = np.sqrt(h_fresa * h_fresa2).sum()
    diff = np.sqrt(h_fresa * h_pina).sum()
    assert same > diff


def test_low_color_pack_still_has_histogram():
    _, hist = preprocess(make_pack("coco"))
    assert hist.sum() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_invalid_image(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)
