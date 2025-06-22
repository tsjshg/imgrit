from PIL import Image
import imgrit

my_image = Image.open("../images/original.jpg")

voronoi_mosaic = imgrit.voronoi_mosaic(my_image, 250)
voronoi_mosaic.save("voronoi-mosaic.png")
warhol_effect = imgrit.warhol_effect(my_image, 10)
warhol_effect.save("warhol-effect.png")
