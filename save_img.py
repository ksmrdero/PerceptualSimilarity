import os

from shutil import copyfile
import cv2
import pylab


# dirs = ['/data/div2k/train/mbt2018-mean-msssim-8','/data/div2k/train/hific-lo', 'heatmaps/mbt', 'heatmaps/hific-lo']
# names = ['mbt.png', 'hific-lo.png', 'mbt_heatmap.png', 'hific-lo_heatmap.png']

# dirs = ['heatmaps/mbt', 'heatmaps/hific-lo']
# names = ['mbt_heatmap.png', 'hific-lo_heatmap.png']
# dest = '/data/div2k/heatmaps_merged'

orig = r'C:\Users\james\Desktop\raw'
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

filenames = os.listdir(orig)
for filename in filenames:
    name = filename.split('.')[0]
    new_folder = os.path.join('merged', name)
    print(filename)

    src = os.path.join(orig, filename)
    dst = os.path.join(new_folder, filename)
    copyfile(src, dst)

    img = cv2.imread(src)
    (success, sal) = saliency.computeSaliency(img)
    pylab.imshow(sal)
    pylab.colorbar()
    pylab.savefig(os.path.join(new_folder, 'saliency.png'), bbox_inches='tight')
    pylab.close()

