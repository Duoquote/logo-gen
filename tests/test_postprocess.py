import numpy as np
from PIL import Image

from logo_gen.postprocess import color_distance, color_key

WHITE = (255, 255, 255)


def test_distance_has_no_overflow():
    rgb = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    dist = color_distance(rgb, WHITE)
    assert not np.isnan(dist).any()
    assert dist[0, 0] == np.sqrt(3 * 255**2)  # black is as far from white as it gets
    assert dist[0, 1] == 0


def test_color_key_clears_white_and_keeps_black():
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (0, 0, 0))
    img.putpixel((1, 0), WHITE)
    out = np.array(color_key(img, WHITE, tolerance=30))
    assert out[0, 0, 3] == 255  # black stays opaque
    assert out[0, 1, 3] == 0  # white becomes transparent


def test_tolerance_edge():
    img = Image.new("RGB", (1, 1), (235, 235, 235))  # about 34.6 away from white
    assert np.array(color_key(img, WHITE, tolerance=10))[0, 0, 3] == 255  # limit 22.1: kept
    assert np.array(color_key(img, WHITE, tolerance=20))[0, 0, 3] == 0  # limit 44.2: cleared
