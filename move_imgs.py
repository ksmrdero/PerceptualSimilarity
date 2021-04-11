import os
from shutil import copy2

# orig_path = '/data/div2k/train/raw_cropped_10px/'
# mbt_path = '/data/div2k/train/mbt2018-mean-msssim-8_cropped_10px/'
# hific_lo_path = '/data/div2k/train/hific_lo_cropped_10px/'
OUTPUT_DIR = 'heatmaps/merged'

mbt_lpips = 'heatmaps/mbt/'
mbt_ssim = 'ssim_heatmaps/mbt/'

hific_lo_lpips = 'heatmaps/hific-lo/'
hific_lo_ssim = 'ssim_heatmaps/hific-lo'


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

filenames = os.listdir(mbt_lpips)
for full_filename in filenames:
    filename = full_filename.split('.')[0]
    new_folder = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    src_mbt_lpips = os.path.join(mbt_lpips, full_filename)
    dest_mbt_lpips = os.path.join(new_folder, filename + '_mbt_lpips.png')
    copy2(src_mbt_lpips, dest_mbt_lpips)

    src_mbt_ssim = os.path.join(mbt_ssim, full_filename)
    dest_mbt_ssim = os.path.join(new_folder, filename + '_mbt_ssim.png')
    copy2(src_mbt_ssim, dest_mbt_ssim)

    src_hific_lo_lpips = os.path.join(hific_lo_lpips, full_filename)
    dest_hific_lo_lpips = os.path.join(
        new_folder, filename + '_hific_lo_lpips.png')
    copy2(src_hific_lo_lpips, dest_hific_lo_lpips)

    src_hific_lo_ssim = os.path.join(hific_lo_ssim, full_filename)
    dest_hific_lo_ssim = os.path.join(
        new_folder, filename + '_hific_lo_ssim.png')
    copy2(src_hific_lo_ssim, dest_hific_lo_ssim)


