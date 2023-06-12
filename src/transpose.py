import PIL
from PIL import Image

image = Image.open("raw_data/slo.png")
image = image.transpose(Image.TRANSPOSE) # WHY?!
image.save("proc_data/slo_t.png")