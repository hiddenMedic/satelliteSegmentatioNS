import numpy as np

in_arr = np.load("proc_data/slo_classes.npy")
print(in_arr.shape)

IMAGE_HEIGHT = in_arr.shape[0]
IMAGE_WIDTH = in_arr.shape[1]
DEPTH = in_arr.shape[2]
FRAME_SIZE = 100
W_FRAMES = IMAGE_WIDTH // FRAME_SIZE
H_FRAMES = IMAGE_HEIGHT // FRAME_SIZE

in_frames = []

def add(x, y, x2, y2):
    offset = FRAME_SIZE // 2
    submat = in_arr[x:x2, y:y2]
    in_frames.append(submat)
    
    if x2 + offset <= IMAGE_HEIGHT:
        submat = in_arr[(x + offset):(x2 + offset), y:y2]
        in_frames.append(submat)

    if y2 + offset <= IMAGE_WIDTH:
        submat = in_arr[x:x2, (y + offset):(y2 + offset)]
        in_frames.append(submat)

    if x2 + offset <= IMAGE_HEIGHT and y2 + offset <= IMAGE_WIDTH:
        submat = in_arr[(x + offset):(x2 + offset), (y + offset):(y2 + offset)]
        in_frames.append(submat)

for i in range(H_FRAMES):
    for j in range(W_FRAMES):
        x = i * FRAME_SIZE
        y = j * FRAME_SIZE
        x2 = x + FRAME_SIZE
        y2 = y + FRAME_SIZE
        add(x, y, x2, y2)
    
    x = i * FRAME_SIZE
    x2 = x + FRAME_SIZE
    y2 = IMAGE_WIDTH
    y = y2 - FRAME_SIZE
    add(x, y, x2, y2)

for j in range(W_FRAMES):
    x2 = IMAGE_HEIGHT
    x = x2 - FRAME_SIZE
    y = j * FRAME_SIZE
    y2 = y + FRAME_SIZE
    add(x, y, x2, y2)
    
x2 = IMAGE_HEIGHT
x = x2 - FRAME_SIZE
y2 = IMAGE_WIDTH
y = y2 - FRAME_SIZE
add(x, y, x2, y2)

in_frames = np.array(in_frames)
print(in_frames.shape)

np.save("proc_data/ovp_y_slo.npy", in_frames)