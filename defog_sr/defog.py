import numpy as np
import math
import cv2


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def adjust_image_gamma(image, gamma=0.98):
    image = np.power(image, gamma)
    max_val = np.max(image.ravel())
    image = image / max_val * 255
    image = image.astype(np.uint8)
    return image


def defog(img, percent=5, adjust_gamma=False, denoise=False):
    assert img.shape[2] == 3
    assert 0 < percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]
        threshold = apply_threshold(channel, low_val, high_val)
        normalized = cv2.normalize(threshold, threshold.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    out_img = cv2.merge(out_channels)
    if adjust_gamma:
        out_img = adjust_image_gamma(out_img)
    if denoise:
        out_img = cv2.fastNlMeansDenoisingColored(out_img, None, 3, 10, 7, 21)

    return out_img
