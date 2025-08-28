from PIL import Image
import imgrit
import logging
import sys

if __name__ == "__main__":
    FORMAT = "%(asctime)s %(message)s"
    logging.basicConfig(format=FORMAT, level="DEBUG")
    logger = logging.getLogger(__name__)
    logger.info("loading images...")
    my_image = Image.open("../images/original.jpg")
    logger.info("creating voronoi mosaic image...")
    voronoi_mosaic = imgrit.voronoi_mosaic(my_image, 250)
    voronoi_mosaic.save("voronoi-mosaic.png")
    logger.info("done")
    logger.info("creating warhol effect image...")
    warhol_effect = imgrit.warhol_effect(my_image, 10)
    warhol_effect.save("warhol-effect.png")
    logger.info("done")
