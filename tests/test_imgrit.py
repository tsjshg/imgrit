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


def _brute_force_boundary_distance(shape, sites):
    h, w = shape
    sites = np.asarray(sites, float)
    yy, xx = np.mgrid[0:h, 0:w]
    p = np.stack([yy, xx], axis=-1).astype(float)[..., None, :]  # (h, w, 1, 2)
    d2 = ((p - sites) ** 2).sum(-1)  # (h, w, n)
    nearest = d2.argmin(-1)
    s1 = sites[nearest]
    with np.errstate(invalid="ignore"):  # 0 / 0 for the nearest site itself
        dist = (d2 - np.take_along_axis(d2, nearest[..., None], -1)) / (
            2 * np.linalg.norm(sites[None, None] - s1[..., None, :], axis=-1)
        )
    dist[np.arange(h)[:, None], np.arange(w)[None, :], nearest] = np.inf
    return nearest, dist.min(-1)


@pytest.mark.parametrize("seed", range(10))
def test_boundary_distance_matches_brute_force(seed):
    """exact distance for few sites (all candidates) and many sites (Delaunay)"""
    from imgrit.imgrit import boundary_distance

    rng = np.random.default_rng(seed)
    n = [3, 10, 40, 80][seed % 4]
    sites = list(dict.fromkeys(map(tuple, rng.integers(0, [40, 60], size=(n, 2)))))
    label, dist = boundary_distance((40, 60), sites)
    label_bf, dist_bf = _brute_force_boundary_distance((40, 60), sites)
    assert np.allclose(dist, dist_bf)
    # pixels equidistant from two sites may get either label, but they are
    # exactly on the boundary
    tie = label != label_bf
    assert np.allclose(dist[tie], 0)


def test_boundary_distance_collinear_sites():
    """Delaunay fails for collinear sites; fall back to all candidates"""
    from imgrit.imgrit import boundary_distance

    sites = [(20, x) for x in range(2, 60, 3)]
    _, dist = boundary_distance((40, 60), sites)
    _, dist_bf = _brute_force_boundary_distance((40, 60), sites)
    assert np.allclose(dist, dist_bf)


def test_boundary_distance_single_site():
    from imgrit.imgrit import boundary_distance

    label, dist = boundary_distance((10, 10), [(5, 5)])
    assert (label == 0).all() and np.isinf(dist).all()


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_voronoi_relief_output(small_img, mode):
    out = imgrit.voronoi_relief(small_img.convert(mode), 10)
    assert out.size == small_img.size and out.mode == "RGB"


def test_voronoi_relief_single_region(small_img):
    out = imgrit.voronoi_relief(small_img, 1)
    assert out.size == small_img.size


def test_relief_image_without_effects_keeps_site_colors(small_img):
    """With no relief, groove, gloss or line, every site keeps its color."""
    from imgrit.imgrit import relief_image

    arr = np.array(small_img)
    sites = [(5, 5), (20, 30), (30, 10), (10, 50)]
    out = np.array(
        relief_image(arr, sites, line_width=0, height=0, groove=0, gloss=0)
    )
    for r, c in sites:
        assert (out[r, c] == arr[r, c]).all()


def test_relief_image_line_color(small_img):
    """A flat relief draws boundary lines exactly in line_color (an int is a gray)."""
    from imgrit.imgrit import relief_image

    arr = np.array(small_img)
    out = np.array(
        relief_image(arr, [(20, 15), (20, 45)], line_width=0.3, line_color=90,
                     height=0, groove=0, gloss=0)
    )
    # the boundary between the two sites is the column x = 30
    assert (out[:, 30] == 90).all()
