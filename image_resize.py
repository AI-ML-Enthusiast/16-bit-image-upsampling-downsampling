# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:38:57 2021

@author: Ibrahim Khalilullah
"""

import cv2
from skimage.transform import resize
from skimage.util import img_as_uint
import skimage.io
from PIL import Image

img_path="depth_00024.png"
resize_width = 1900
resize_height = 1400

################# using opencv-python ######################
'''
opencv-python interpolation methods
INTER_NEAREST – a nearest-neighbor interpolation INTER_LINEAR – a bilinear interpolation 
(used by default) INTER_AREA – resampling using pixel area relation. It may be a preferred method 
for image decimation, as it gives moire’-free results. But when the image is zoomed, it is 
similar to the INTER_NEAREST method. INTER_CUBIC – a bicubic interpolation over 4×4 pixel 
neighborhood INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood
'''

img=cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
image_resize=cv2.resize(img, (resize_width, resize_height), cv2.INTER_CUBIC)
cv2.imwrite("image_resize_cubic_opencv.png", image_resize)

###############  using scikit-image resize function with different interpolation method ##########

'''
The order of interpolation. The order has to be in the range 0-5:
    
0: Nearest-neighbor

1: Bi-linear (default)

2: Bi-quadratic

3: Bi-cubic

4: Bi-quartic

5: Bi-quintic
'''

img=img_as_uint(skimage.io.imread(img_path))
image_resize_biquartic=resize(img, (resize_width, resize_height), order=4, anti_aliasing=True)
skimage.io.imsave("image_resize_biquartic_scikit.png", img_as_uint(image_resize_biquartic))

##########################  using PIL ########################################

img=Image.open(img_path)
img_resize = img.resize((resize_width, resize_height), Image.BOX)

img_resize.save("image_resize_box_pil.png")