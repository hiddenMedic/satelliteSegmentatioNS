import numpy as np
import PIL
from PIL import Image

FRAME_SIZE = 100
IMAGE_WIDTH = 1980
IMAGE_HEIGHT = 2828
H_FRAMES = IMAGE_HEIGHT // FRAME_SIZE
W_FRAMES = IMAGE_WIDTH // FRAME_SIZE
# he = 2828, wid = 1980, taken from the numpy array dimensions height = shape[0], width = shape[1]

frames = np.load("proc_data/ns_classes_frames.npy")
print(frames.shape)

out_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (250,250,250))

colors = [(5, 73, 7), 
 (6, 154, 243),
 (128, 96, 0), 
 (149, 208, 252), 
 (166, 166, 166), 
 (220, 20, 60), 
 (255, 165, 0), 
 (255, 255, 0), 
 (255, 255, 255)]

pixels = out_image.load()

def paste(x, y, frame): # x - height, y - width
    # print(x, y)
    for i in range(FRAME_SIZE):
        for j in range(FRAME_SIZE):
            corva = np.argmax(frame[i, j])
            clr = (0, 0, 0) if corva >= 9 else colors[corva]
            pixels[y + j, x + i] = clr

idx = 0

for i in range(H_FRAMES):
    for j in range(W_FRAMES):
        x = i * FRAME_SIZE
        y = j * FRAME_SIZE
        paste(x, y, frames[idx])
        idx += 1
    
    x = i * FRAME_SIZE
    y = IMAGE_WIDTH - FRAME_SIZE
    paste(x, y, frames[idx])
    idx += 1

for j in range(W_FRAMES):
    x = IMAGE_HEIGHT - FRAME_SIZE
    y = j * FRAME_SIZE
    paste(x, y, frames[idx])
    idx += 1

x = IMAGE_HEIGHT - FRAME_SIZE
y = IMAGE_WIDTH - FRAME_SIZE
paste(x, y, frames[idx])
idx += 1

out_image = out_image.transpose(Image.TRANSPOSE) # WHY?!
out_image.save("figures/novi_sad_76_aug.png")


# pixels[i, j] -> i = width, j = height
# frame[i, j] -> i = height, j = width