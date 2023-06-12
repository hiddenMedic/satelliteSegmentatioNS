import random
import numpy as np

TRANS_PER_FRAME = 2

transformations = [
    "12", "21", "22", "31", "32", "41", "42"
] # "11" is nothing

in_frames = np.load("proc_data/ovp_x_slo.npy")
out_frames = np.load("proc_data/ovp_y_slo.npy")
aug_in_frames = []
aug_out_frames = []

for in_frame, out_frame in zip(in_frames, out_frames):
    aug_in_frames.append(in_frame)
    aug_out_frames.append(out_frame)

    transs = random.sample(transformations, 2)
    for trans in transs:
        rot = int(trans[0]) - 1
        flip = int(trans[1]) - 1
        new_in_frame = in_frame
        new_out_frame = out_frame
        for i in range(rot):
            new_in_frame = np.rot90(new_in_frame)
            new_out_frame = np.rot90(new_out_frame)
        if flip:
            new_in_frame = np.fliplr(new_in_frame)
            new_out_frame = np.fliplr(new_out_frame)
        aug_in_frames.append(new_in_frame)
        aug_out_frames.append(new_out_frame)

aug_in_frames = np.array(aug_in_frames)
aug_out_frames = np.array(aug_out_frames)
print(aug_in_frames.shape, aug_out_frames.shape)

np.save("proc_data/aug_x_slo.npy", aug_in_frames)
np.save("proc_data/aug_y_slo.npy", aug_out_frames)