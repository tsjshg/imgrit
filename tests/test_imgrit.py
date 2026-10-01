"""Regression tests for imgrit.

Run with: pytest tests/
"""

import numpy as np
import pytest
from PIL import Image
from scipy.spatial import Voronoi

import imgrit
from imgrit.imgrit import KMeansImage, VoronoiImage, find_edge_point


@pytest.fixture
def small_img():
    """A small RGB image with some structure."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_version_is_available():
    assert isinstance(imgrit.__version__, str)


def test_sites_as_ndarray(small_img):
    """VoronoiImage must accept a NumPy array of sites (docstring: array-like)."""
    sites = np.array([[5, 5], [20, 20], [30, 10]])
    vi = VoronoiImage(small_img, sites)
    assert vi.sites_num == 3


def test_sites_out_of_frame_raises(small_img):
    with pytest.raises(ValueError):
        VoronoiImage(small_img, [(5, 5), (100, 100)])


def test_grayscale_input(small_img):
    """L-mode images used to crash on shape unpacking."""
    out = imgrit.voronoi_mosaic(small_img.convert("L"), 10)
    assert out.size == small_img.size


def test_rgba_input(small_img):
    out = imgrit.voronoi_mosaic(small_img.convert("RGBA"), 10)
    assert out.size == small_img.size


def test_random_mode_sites_stay_in_frame(small_img):
    """Edge pixels picked in random mode must map back inside the frame."""
    for _ in range(5):
        out = imgrit.voronoi_mosaic(small_img, 30, random=True)
        assert out.size == small_img.size


def test_invalid_mode_raises(small_img):
    with pytest.raises(ValueError):
        imgrit.voronoi_mosaic(small_img, 5, mode="colour")


def test_warhol_preserves_dim_colors():
    """Cluster colors must be scaled back with the factors used for scaling,
    not blindly with 255 (dim images used to come out too bright)."""
    arr = np.zeros((30, 30, 3), np.uint8)
    arr[:15] = [100, 50, 25]
    arr[15:] = [90, 40, 20]
    out = np.array(imgrit.warhol_effect(Image.fromarray(arr), 2))
    in_max = arr.reshape(-1, 3).max(axis=0)
    out_max = out.reshape(-1, 3).max(axis=0)
    # allow +1 for rounding
    assert (out_max <= in_max + 1).all()


def test_warhol_on_black_image():
    """A constant-zero channel used to cause division by zero (NaN)."""
    black = Image.fromarray(np.zeros((20, 20, 3), np.uint8))
    out = np.array(imgrit.warhol_effect(black, 2))
    assert (out == 0).all()


def test_find_edge_point_vertical_bisector():
    """The bisector of two sites sharing a row is vertical: both endpoints
    of the drawn segment must share the same x coordinate."""
    pts = [(50, 5), (50, 25), (10, 15)]  # (row, col)
    vor = Voronoi(pts)
    checked = 0
    for k, v in vor.ridge_dict.items():
        if -1 in v and set(k) == {0, 1}:
            edge_point, vertex = find_edge_point(v, k, vor, x_max=40, y_max=60)
            assert edge_point[0] == pytest.approx(vertex[0])
            checked += 1
    assert checked == 1


def test_duplicated_centroids_are_dropped(small_img):
    """Rounded k-means centroids may collide; duplicated sites must not
    reach scipy's Voronoi (QhullError)."""
    # more sites than a 4x4 thumbnail can hold distinct positions for
    tiny = KMeansImage(small_img.resize((4, 4)))
    out, _ = tiny.voronoi_img(10, boundary=True)
    assert out.size == (4, 4)


def test_kmeans_sample_size():
    """Sample size grows with the number of sites, not with the image size,
    and never exceeds the number of pixels."""
    from imgrit.imgrit import kmeans_sample_size

    assert kmeans_sample_size(10_000_000, 20) == 10_000  # lower bound
    assert kmeans_sample_size(10_000_000, 250) == 50_000  # 200 per site
    assert kmeans_sample_size(10_000_000, 1000) == 200_000
    assert kmeans_sample_size(1_000_000, 250) == kmeans_sample_size(10_000_000, 250)
    assert kmeans_sample_size(2_400, 250) == 2_400  # capped by pixels


def test_subsample_for_kmeans():
    from imgrit.imgrit import subsample_for_kmeans

    data = np.arange(300_000 * 3, dtype=float).reshape(-1, 3)
    sub = subsample_for_kmeans(data, 250)
    assert sub.shape == (50_000, 3)
    # rows are picked without replacement from the original data
    assert len(np.unique(sub[:, 0])) == 50_000
    assert np.isin(sub[:, 0], data[:, 0]).all()
    # small data is returned as is
    small = data[:5_000]
    assert subsample_for_kmeans(small, 250) is small
