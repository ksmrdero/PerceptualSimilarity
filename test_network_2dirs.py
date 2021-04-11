# python test_network_2dirs.py -p /data/div2k/train/raw_cropped_10px/ -t1 /data/div2k/train/mbt2018-mean-msssim-8_cropped_10px/ -t2 /data/div2k/train/hific_lo_cropped_10px/
# python test_network_2dirs.py -p /data/div2k/train/raw/ -t1 /data/div2k/train/jpg-png/ -t2 /data/div2k/train/hific-hi/
import torch
import lpips
import argparse
import os
from IPython import embed

use_gpu = False         # Whether to use GPU
spatial = True         # Return a spatial map of perceptual distance.

OUTPUT_DIR = 'heatmaps'
LOG_DIR = 'logs'
OUTPUT1 = 'mbt'
OUTPUT2 = 'hific-lo'
LOG_NAME = '%s_%s.csv'%(OUTPUT1, OUTPUT2)


# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

if(use_gpu):
	loss_fn.cuda()

def get_map(orig_path, test_path1, test_path2, filename):

    ex_ref = lpips.im2tensor(lpips.load_image(orig_path))
    ex_p0 = lpips.im2tensor(lpips.load_image(test_path1))
    ex_p1 = lpips.im2tensor(lpips.load_image(test_path2))

    if(use_gpu):
        ex_ref = ex_ref.cuda()
        ex_p0 = ex_p0.cuda()
        ex_p1 = ex_p1.cuda()

    ex_d0 = loss_fn.forward(ex_ref,ex_p0)
    ex_d1 = loss_fn.forward(ex_ref,ex_p1)

    if not spatial:
        print('Distances: (%.3f, %.3f)'%(ex_d0, ex_d1))
    else:
        print('Distances: (%.3f, %.3f)'%(ex_d0.mean(), ex_d1.mean()))            # The mean distance is approximately the same as the non-spatial distance
        
        # Visualize a spatially-varying distance map between ex_p0 and ex_ref
        import pylab
        pylab.imshow(ex_d0[0,0,...].data.cpu().numpy())
        # pylab.show()
        pylab.colorbar()
        pylab.savefig(os.path.join(OUTPUT_DIR, OUTPUT1, filename), bbox_inches='tight')
        pylab.close()

        pylab.imshow(ex_d1[0, 0, ...].data.cpu().numpy())
        # pylab.show()
        pylab.colorbar()
        pylab.savefig(os.path.join(OUTPUT_DIR, OUTPUT2,
                                   filename), bbox_inches='tight')
        pylab.close()

    return(ex_d0.mean(), ex_d1.mean())


def process_files(args):
    filenames = os.listdir(args.orig_dir)

    sim = open(os.path.join(LOG_DIR, LOG_NAME), 'w')
    sim.write('name,im1,im2\n')

    for idx, filename in enumerate(filenames):
        orig_path = os.path.join(args.orig_dir, filename)
        test_path1 = os.path.join(args.test_dir1, filename)
        test_path2 = os.path.join(args.test_dir2, filename)

        d1, d2 = get_map(orig_path, test_path1, test_path2, filename)
        sim.write('%s,%.3f,%.3f\n'%(filename, d1, d2))
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
