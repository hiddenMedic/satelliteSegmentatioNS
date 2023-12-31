﻿# satelliteSegmentatioNS

**Training data:** <br>
**Input:** 13 band image of Slovenia, here is a 3 band (rgb) example: <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/rgb_slo.png?raw=true" width="300" height="300"> <br>
**Output:** 3 band (rgb) image of Slovenia colored with 9 classes according to the legend: <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/slo.png?raw=true" width="300" height="300"> <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/legend.png?raw=true"> <br> <br>

**Architecture:**
Image is split into 100x100 frames. Extra frames that overlap are taken to increase data size. Augmentation may be done. If augmented, for each frame 2 random additional transformations are performed (out of the possible 7, from D<sub>4</sub> without the identity). The U-Net architecture is used. <br> <br>

**After training the model on the data from Slovenia, we perform the semantic segmentation on Novi Sad:** <br><br>
Using only overlapping, 160 epochs. Accuracy 71%. <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/figures/u-net-160-epoch-no-aug.png?raw=true"> <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/figures/novi_sad_71_overlap.png?raw=true" width="564" height="396"> <br> <br>

Using only overlapping, 60 epochs. Accuracy 81%. <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/figures/u-net-60-epoch-no-aug.png?raw=true"> <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/figures/novi_sad_81_overlap.png?raw=true" width="564" height="396"> <br> <br>

Using overlapping and augmentation, 160 epochs. Accuracy 76%. <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/figures/u-net-160-epoch-aug.png?raw=true"> <br>
<img src="https://github.com/hiddenMedic/satelliteSegmentatioNS/blob/main/figures/novi_sad_76_aug.png?raw=true"  width="564" height="396">
