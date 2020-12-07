import numpy as np
import torch
import torchvision.transforms.functional as F

class LongSideResize:
    """
    Resizes the input such that the longest side is equal to
    width or height (depending on whether the width or height is longest).
    """
    def __init__(self, width, height):
        self.target_width = width
        self.target_height = height

    def __call__(self, input):
        """
        :param input: a PIL image
        :return:
        """
        h, w, c = np.array(input).shape

        if h > w:
            resize_factor = self.target_height / h
            resized_width = int(w*resize_factor)
            resized_height = self.target_height
            if resized_width > self.target_width:
                raise Exception("resized_width > target_width")

        else:
            resize_factor = self.target_width / w
            resized_height = int(h * resize_factor)
            resized_width = self.target_width

            if resized_height > self.target_height:
                raise Exception("resized_height > target_height")

        return F.resize(input, (resized_height, resized_width))


class Pad:
    """
    pads the image with zeros such that the final size is (height, width).
    """
    def __init__(self, width, height, return_mask=False):
        self.target_width = width
        self.target_height = height
        self.return_mask = return_mask

    def __call__(self, input):
        """
        :param input: a torch Tensor
        :return:
        """
        c, h, w = np.array(input).shape

        if h > w:
            resize_factor = self.target_height/h
            resized_width = int(w*resize_factor)
            resized_height = self.target_height
            if resized_width > self.target_width:
                raise Exception("resized_width > target_width")

            total_width_padding = self.target_width - resized_width
            left_padding = total_width_padding // 2
            right_padding = total_width_padding // 2

            if total_width_padding%2 == 1:
                right_padding += 1

            padding = (left_padding, 0, right_padding, 0)
        else:
            resize_factor = self.target_width / w
            resized_height = int(h * resize_factor)
            resized_width = self.target_width

            if resized_height > self.target_height:
                raise Exception("resized_height > target_height")

            total_width_padding = self.target_height - resized_height
            top_padding = total_width_padding // 2
            bottom_padding = total_width_padding // 2

            if total_width_padding % 2 == 1:
                bottom_padding += 1

            padding = (0, top_padding, 0, bottom_padding)

        new_input = torch.zeros((3, self.target_height, self.target_width), dtype=torch.float32)
        new_input[:, padding[1]:padding[1]+resized_height, padding[0]:padding[0]+resized_width] = input

        if not self.return_mask:
            return new_input

        mask = torch.zeros((3, self.target_height, self.target_width), dtype=torch.float32)
        mask[:, padding[1]:padding[1]+resized_height, padding[0]:padding[0]+resized_width] = 1

        return new_input, mask


class ToNumpy:

    def __call__(self, input):
        return np.array(input, dtype=torch.float32)


class ToTensor:

    def __call__(self, input):
        return F.to_tensor(input)


class ImageNetNormalize:

    def __call__(self, input):
        """
        :param input: a torch Tensor
        :return:
        """
        return F.normalize(
            input,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            inplace=True
        )
