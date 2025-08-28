# README

This is a tiny image processing library to conver your images to Voronoi mosaic or Warhol effect images. We used k-means clustering algorithms to determine the position of Voronoi sites and pixel groups of Warhol effect.

# How to use

`pip install imgrit`

The library depends on [Pillow](https://pypi.org/project/pillow/), [NumPy](https://pypi.org/project/numpy/), and [SciPy](https://pypi.org/project/scipy/).

If you have [scikit-learn](https://scikit-learn.org/stable/), the library uses the faster k-means.

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


## OpenSea

If you'd like to see more images, please visit [Asakura Gallery Digital](https://opensea.io/collection/asakura) at OpenSea.

## Citations

under preparetion.