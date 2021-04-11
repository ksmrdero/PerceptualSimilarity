# pip install opencv_contrib_python
import cv2
import numpy as np
import torch
import lpips
import argparse
import os
from IPython import embed

use_gpu = False         # Whether to use GPU
orig_path = 'orig_0136_mbt.png'
test_path = '0136_lo_cropped.png'

loss_fn = lpips.Saliency_LPIPS(net='alex')
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

if(use_gpu):
    loss_fn.cuda()

saliency = cv2.saliency.StaticSaliencySpectralResidual_create()


ex_ref = lpips.load_image(orig_path)
ex_p0 = lpips.load_image(test_path)

if(use_gpu):
    ex_ref = ex_ref.cuda()
    ex_p0 = ex_p0.cuda()

# print(saliency.computeSaliency(lpips.load_image(orig_path)))

ex_d0 = loss_fn.forward(ex_ref, ex_p0)
# img = cv2.imread(orig_path)
# (success, sal) = saliency.computeSaliency(img)

# score = (ex_d0.detach().numpy() * (sal)).mean()

print(ex_d0)