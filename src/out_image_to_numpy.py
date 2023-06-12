from PIL import Image
import numpy as np

image = Image.open("proc_data/slo_t.png")
 
NUM_CLASSES = 9

# summarize some details about the image
print(image.format)
width, height = image.size
print(image.size)
print(image.mode)
numpydata = np.asarray(image)
print(numpydata.shape)

# np.save("proc_data/rgb_slo.npy", numpydata)

pix = image.load()
classes = set()

for x in range(width):
    for y in range(height):
        r, g, b = pix[x, y]
        classes.add((r, g, b))

classes = list(classes)
classes = sorted(classes)

out_classed = np.zeros(shape=(height, width))

for x in range(width):
    for y in range(height):
        r, g, b = pix[x, y]
        out_classed[x, y] = classes.index((r, g, b))

# print(out_classed)
print(out_classed.shape)
print(classes)
print(len(classes))
# np.save("proc_data/out_classes.npy", out_classed)

out_one_hot = np.zeros(shape=(height, width, NUM_CLASSES))

for x in range(width):
    for y in range(height):
        val = int(out_classed[x, y])
        out_one_hot[x, y, val] = 1

print(out_one_hot.shape)
np.save("proc_data/slo_classes.npy", out_one_hot)

# print(out_one_hot)