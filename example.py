import os
import PIL.Image as Image
import numpy as np

tempwidth = 2000
templength = 2000
for filename in os.listdir("../Data/small/trainingData/spectrograms"):
    if filename.endswith(".png"):
        img = Image.open("../Data/small/trainingData/spectrograms/" + filename)
        length,width = np.array(img).shape
        if templength > length:
            templength = length
        if tempwidth > width:
            tempwidth = width
    else:
        continue

print templength, tempwidth
