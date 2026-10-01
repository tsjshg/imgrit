# README

This is a tiny image processing library to convert your images to Voronoi mosaic or Warhol effect images. We used k-means clustering algorithms to determine the position of Voronoi sites and pixel groups of Warhol effect.

# How to use

`pip install imgrit`

The library depends on [Pillow](https://pypi.org/project/pillow/), [NumPy](https://pypi.org/project/numpy/), and [SciPy](https://pypi.org/project/scipy/).

If you have [scikit-learn](https://scikit-learn.org/stable/), the library uses the faster k-means. You can install it together with `pip install "imgrit[sklearn]"`.

The following is the input image.

<img width="50%" src="https://github.com/tsjshg/imgrit/blob/main/images/original.jpg?raw=true">

```python
from PIL import Image
import imgrit

my_image = Image.open("../images/original.jpg")
voronoi_mosaic = imgrit.voronoi_mosaic(my_image, 250)
voronoi_mosaic.save("voronoi-mosaic.png")
```

<img width="50%" src="https://github.com/tsjshg/imgrit/blob/main/images/voronoi-mosaic.png?raw=true">

```
warhol_effect = imgrit.warhol_effect(my_image, 10)
warhol_effect.save("warhol-effect.png")
```

<img width="50%" src="https://github.com/tsjshg/imgrit/blob/main/images/warhol-effect.png?raw=true">

## Performance

Rough execution times by image size. `voronoi_mosaic` fits k-means on a random subsample of pixels (200 pixels per region, at least 10,000 pixels), so the number of regions has little effect on the time up to a few hundred regions. For large images, most of the time is spent on pixel-wise processing, which grows with the number of pixels.

With scikit-learn (`pip install "imgrit[sklearn]"`):

| Image size | `voronoi_mosaic`<br>20 regions (default) | `voronoi_mosaic`<br>250 regions | `voronoi_mosaic`<br>1000 regions | `warhol_effect`<br>5 colors (default) | `warhol_effect`<br>10 colors |
|---|---|---|---|---|---|
| 0.5 MP (800×600) | 0.6 s | 1.0 s | 8.7 s | 0.4 s | 0.5 s |
| 1.9 MP (1600×1200) | 2.5 s | 3.0 s | 10 s | 1.6 s | 2.1 s |
| 7.7 MP (3200×2400) | 9.7 s | 10 s | 18 s | 6.0 s | 7.4 s |
| 12.2 MP (4032×3024) | 16 s | 17 s | 25 s | 9.9 s | 12 s |

Without scikit-learn (SciPy's k-means is used):

| Image size | `voronoi_mosaic`<br>20 regions (default) | `voronoi_mosaic`<br>250 regions | `voronoi_mosaic`<br>1000 regions | `warhol_effect`<br>5 colors (default) | `warhol_effect`<br>10 colors |
|---|---|---|---|---|---|
| 0.5 MP (800×600) | 0.6 s | 2.8 s | 147 s | 0.4 s | 0.5 s |
| 1.9 MP (1600×1200) | 2.5 s | 5.0 s | 149 s | 1.7 s | 2.0 s |
| 7.7 MP (3200×2400) | 10 s | 13 s | 163 s | 7.1 s | 8.3 s |
| 12.2 MP (4032×3024) | 16 s | 18 s | 163 s | 11 s | 13 s |

SciPy's k-means becomes very slow with many regions, so installing scikit-learn is recommended if you use more than a few hundred regions. Peak memory usage was about 4 GB for a 12 MP image.

Measurement environment: Mac mini (Apple M4, 10 cores, 24 GB memory), macOS 15.7, Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, Pillow 12.3.0, scikit-learn 1.9.1. Each value is a single run on `images/original.jpg` resized to the given size. Times vary with your machine and the input image, so please use them as a rough guide.


## OpenSea

If you'd like to see more images, please visit [Asakura Gallery Digital](https://opensea.io/collection/asakura) at OpenSea.

## Citations

[A novel method for Voronoi mosaic effect using k-means clustering](https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/5152/) 
