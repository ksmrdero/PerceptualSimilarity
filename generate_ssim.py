# python generate_ssim.py -p /data/div2k/train/raw_cropped_10px/ -t1 /data/div2k/train/mbt2018-mean-msssim-8_cropped_10px/ -t2 /data/div2k/train/hific_lo_cropped_10px/
from skimage.measure import compare_ssim
import cv2
import numpy as np
import os
import argparse
import pylab

OUTPUT_DIR = 'ssim_heatmaps'
LOG_DIR = 'logs'
OUTPUT1 = 'mbt'
OUTPUT2 = 'hific-lo'
LOG_NAME = '%s_%s_ssim.csv' % (OUTPUT1, OUTPUT2)

def write_diff(before, after):
    im1 = before.copy()
    im2 = after.copy()
    # Convert images to grayscale
    before_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = compare_ssim(before_gray, after_gray, full=True)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    # diff = (diff * 255).astype("uint8")

    return diff, score


def process_files(args):
    filenames = os.listdir(args.orig_dir)
    sim = open(os.path.join(LOG_DIR, LOG_NAME), 'w')
    sim.write('name,im1,im2\n')

    for idx, filename in enumerate(filenames):
        
        orig_path = os.path.join(args.orig_dir, filename)
        test_path1 = os.path.join(args.test_dir1, filename)
        test_path2 = os.path.join(args.test_dir2, filename)

        orig = cv2.imread(orig_path)
        im1 = cv2.imread(test_path1)
        im2 = cv2.imread(test_path2)

        diff1, score1 = write_diff(orig, im1)
        diff2, score2 = write_diff(orig, im2)
        pylab.imshow(1 - diff1)
        # pylab.show()
        pylab.colorbar()
        pylab.savefig(os.path.join(OUTPUT_DIR, OUTPUT1,
                                   filename), bbox_inches='tight')
        pylab.close()

        pylab.imshow(1 - diff2)
        # pylab.show()
        pylab.colorbar()
        pylab.savefig(os.path.join(OUTPUT_DIR, OUTPUT2,
                                   filename), bbox_inches='tight')
        pylab.close()

        print(filename, score1, score2)
        sim.write('%s,%.3f,%.3f\n' % (filename, score1, score2))
    sim.close()


def main(**kwargs):
    description = "runs 2afc test for directory"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--orig_dir", type=str,
                        required=True, help="path to directory containing original images")
    parser.add_argument("-t1", "--test_dir1", type=str, required=True,
                        help="path to directory containing test images 1")
    parser.add_argument("-t2", "--test_dir2", type=str, required=True,
                        help="path to directory containing test images 2")
    args = parser.parse_args()
    process_files(args)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT1)):
        os.makedirs(os.path.join(OUTPUT_DIR, OUTPUT1))
    if not os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT2)):
        os.makedirs(os.path.join(OUTPUT_DIR, OUTPUT2))
    main()
