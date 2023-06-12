import numpy as np

in_arr = np.load("raw_data/in_ns.npy")
print(in_arr.shape)

IMAGE_HEIGHT = in_arr.shape[0]
IMAGE_WIDTH = in_arr.shape[1]
IN_DEPTH = 11 # in_arr.shape[2]
FRAME_SIZE = 100
W_FRAMES = IMAGE_WIDTH // FRAME_SIZE
H_FRAMES = IMAGE_HEIGHT // FRAME_SIZE

in_frames = []

for i in range(H_FRAMES):
    for j in range(W_FRAMES):
        x = i * FRAME_SIZE
        y = j * FRAME_SIZE
        x2 = x + FRAME_SIZE
        y2 = y + FRAME_SIZE
        submat = in_arr[x:x2, y:y2]
        in_frames.append(submat)
    
    # assuming IMAGE_WIDTH % FRAME_SIZE != 0, its fine if this is not true
    x = i * FRAME_SIZE
    x2 = x + FRAME_SIZE
    y2 = IMAGE_WIDTH
    y = y2 - FRAME_SIZE
    submat = in_arr[x:x2, y:y2]
    in_frames.append(submat)

 # assuming IMAGE_HEIGHT % FRAME_SIZE != 0, its fine if this is not true
for j in range(W_FRAMES):
    x2 = IMAGE_HEIGHT
    x = x2 - FRAME_SIZE
    y = j * FRAME_SIZE
    y2 = y + FRAME_SIZE
    submat = in_arr[x:x2, y:y2]
    in_frames.append(submat)

# assuming IMAGE_WIDTH % FRAME_SIZE != 0, its fine if this is not true
x2 = IMAGE_HEIGHT
x = x2 - FRAME_SIZE
y2 = IMAGE_WIDTH
y = y2 - FRAME_SIZE
submat = in_arr[x:x2, y:y2]
in_frames.append(submat)

in_frames = np.array(in_frames)
print(in_frames.shape)

np.save("proc_data/in_ns_frames.npy", in_frames)