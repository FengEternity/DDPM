import torch
import torch.nn.functional as F
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_float

class DarkChannel:
    @staticmethod
    def min_box(image, kernel_size=15):
        selem = disk(kernel_size)
        min_image = rank.minimum(image, selem)
        return min_image

    @staticmethod
    def calculate_dark(image, window_size):
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input image is not a torch.Tensor")

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Input image should have shape [batch_size, 3, height, width]")

        image = torch.clamp(image, 0, 1)  # 将像素值裁剪到 [0, 1] 范围内

        dark = torch.min(image[:, 0, :, :], dim=1).values
        dark = DarkChannel.min_box(dark, kernel_size=window_size)

        dark = dark / dark.max()

        return dark


input_image = torch.randn(32, 3, 64, 64)
window_size = 15
output = DarkChannel.calculate_dark(input_image, window_size)
print("output.shape", output.shape)