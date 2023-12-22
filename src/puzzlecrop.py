"""
Write custom transform as torch nn.Module
"""

import torch
import torch.nn as nn
import numpy as np
import numbers
from PIL import Image
import torchvision.transforms as transforms

class Puzzlecrop(nn.Module):
    def __init__(self, size, stride):  # stride == n
        super().__init__()

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.stride = stride

    # @torch.no_grad()
    def get_params(self, img, output_size, stride):

        if self._is_pil_image(img):
            w, h = img.size
        elif isinstance(img, torch.Tensor):

            h, w = img.size()[-2:]
        else:
            raise TypeError(
                "img should be PIL Image or PyTorch tensor. Got {}".format(type(img))
            )

        # w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = np.sort(
            list(range(0, h, th)) + list(range(stride, h - th, th))
        ).tolist()
        j_list = np.sort(
            list(range(0, w, tw)) + list(range(stride, w - tw, tw))
        ).tolist()

        return i_list, j_list, th, tw

    def _is_pil_image(self, img):
        return isinstance(img, Image.Image)

    # img, i_list, j_list, th, tw
    def custom_crops(self, img, x, y, th, tw):

        crops = []
        for iii in range(len(x)):
            for jjj in range(len(y)):

                crop_coord = (y[jjj], x[iii], y[jjj] + tw, x[iii] + th)

                if self._is_pil_image(img):
                    new_crop = img.crop(crop_coord)
                elif isinstance(img, torch.Tensor):

                    new_crop = transforms.functional.crop(
                        img, top=y[jjj], left=x[iii], height=tw, width=th
                    )
                else:
                    raise TypeError(
                        "img should be PIL Image or PyTorch tensor. Got {}".format(
                            type(img)
                        )
                    )
                crops.append(new_crop)

        return tuple(crops)

    @torch.no_grad()
    def forward(self, x):
        i, j, th, tw = self.get_params(x, self.size, self.stride)
        return self.custom_crops(x, i, j, th, tw)
