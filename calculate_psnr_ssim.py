import os
import cv2
import eval_diffusion
from utils.metrics import calculate_psnr, calculate_ssim


config = eval_diffusion.config_get()
gt_path = os.path.join(config.data.test_data_dir, 'target/')
results_path = config.data.test_save_dir

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))

cumulative_psnr, cumulative_ssim = 0, 0
for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)))
print(results_path)
