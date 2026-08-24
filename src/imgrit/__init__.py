from importlib.metadata import PackageNotFoundError, version

from .imgrit import voronoi_mosaic
from .imgrit import warhol_effect

try:
    __version__ = version(__package__)
except PackageNotFoundError:
    # running from a source tree without installation
    __version__ = "0+unknown"
