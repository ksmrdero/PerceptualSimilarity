import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

#  python saliency.py -p /data/div2k/train/raw_cropped_10px/ -t1 /data/div2k/train/mbt2018-mean-msssim-8_cropped_10px/

# python saliency.py -p /data/div2k/train/raw/ -t1 /data/div2k/train/
# python saliency.py -p /data/coco/val/val2017/ -t1 /data/coco/val/
# pip install opencv_contrib_python
import torch
import lpips
import argparse
import os
from IPython import embed

use_gpu = True         # Whether to use GPU
use_folder = True       # multiple folder runs (true) or just single

LOG_DIR = 'logs'


def process_files(args):
    # 'hific-hi', 'hific-lo', 'bmshj2018-hyperprior-msssim-8', 'bmshj2018-hyperprior-msssim-1', 'jpg-png','mbt2018-mean-msssim-1', 'mbt2018-mean-msssim-8'
    folders = ['hific-hi', 'hific-lo']
    # folders = ['mbt']  
    for folder in folders:
        LOG_NAME = '%s_saliency_gauss_filter_1-.csv' % (folder)

        filenames = os.listdir(args.orig_dir)
        if use_folder:
            test_dir = os.path.join(args.test_dir, folder)
        else:
            test_dir = args.test_dir

        # writing files
        sim = open(os.path.join(LOG_DIR, LOG_NAME), 'w')
        sim.write('name,lpips,weighted_score\n')

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        for idx, filename in enumerate(filenames):
            orig_path = os.path.join(args.orig_dir, filename)
            test_path = os.path.join(test_dir, filename)

            ex_ref = lpips.im2tensor(lpips.load_image(orig_path))
            try:

                ex_p0 = lpips.im2tensor(lpips.load_image(test_path))
            except:
                continue

            if(use_gpu):
                ex_ref = ex_ref.cuda()
                ex_p0 = ex_p0.cuda()

            ex_d0 = loss_fn.forward(ex_ref, ex_p0)
            img = cv2.imread(orig_path)
            (success, sal) = saliency.computeSaliency(img)
            # print()

            #using gaussian filter
            sal_filtered = gaussian_filter(sal, sigma=5)
            # print(sal_filtered)
            if use_gpu:
                score = (ex_d0.cpu().detach().numpy()
                         * (sal_filtered)).mean()
            else:
                score = (ex_d0.detach().numpy() * (sal_filtered)).mean()

            sim.write('%s,%.4f,%.8f\n' % (filename, ex_d0.mean(), score))
            print('%s, %.4f, %.8f' % (filename, ex_d0.mean(), score))

        sim.close()


def main(**kwargs):
    description = "runs 2afc test for directory"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--orig_dir", type=str,
                        required=True, help="path to directory containing original images")
    parser.add_argument("-t1", "--test_dir", type=str, required=True,
                        help="path to directory containing test images 1")
    args = parser.parse_args()
    process_files(args)


if __name__ == '__main__':

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Linearly calibrated models (LPIPS)
    # Can also set net = 'squeeze' or 'vgg'
    loss_fn = lpips.LPIPS(net='alex', spatial=True)
    # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

    if(use_gpu):
        loss_fn.cuda()

    main()
