import numpy as np

x = np.load("proc_data/ovp_rgb_slo.npy")
y = np.load("proc_data/ovp_y_slo.npy")

colors = [(5, 73, 7), 
 (6, 154, 243),
 (128, 96, 0), 
 (149, 208, 252), 
 (166, 166, 166), 
 (220, 20, 60), 
 (255, 165, 0), 
 (255, 255, 0), 
 (255, 255, 255)]

print(x.shape)
print(y.shape)

for (a, b) in zip(x, y):  
    for (i1, i2) in zip(a, b):
        for (j1, j2) in zip(i1, i2):
            thec = colors[np.argmax(j2)]
            if np.all(j1 == thec) != True:
                print(list(j1), j2, list(thec))
                assert(0)
                
                
# BIG PROBLEM