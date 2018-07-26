import numpy as np
import cv2
import time


def gen_mask(flags):
    np.random.seed(seed=int(time.time()))  # set random seed according to the current time
    masks = np.ones((flags.sample_batch, flags.img_size, flags.img_size), dtype=np.float32)

    if flags.mask_type == 'center':
        scale = 0.25
        low, upper = int(flags.img_size * scale), int(flags.img_size * (1.0 - scale))
        masks[:, low:upper, low:upper] = 0.
    elif flags.mask_type == 'random':
        ratio = 0.8
        masks[np.random.random((flags.sample_batch, flags.img_size, flags.img_size)) <= ratio] = 0.
    elif flags.mask_type == 'half':
        half_types = np.random.randint(4, size=flags.sample_batch)
        masks = [half_mask(half_types[idx], flags.img_size) for idx in range(flags.sample_batch)]
        masks = np.asarray(masks)
    elif flags.mask_type == 'pattern':
        masks = [pattern_mask(flags.img_size) for _ in range(flags.sample_batch)]
        masks = np.asarray(masks)
    else:
        raise NotImplementedError

    return masks


def half_mask(half_type, img_size):
    mask = np.ones((img_size, img_size), dtype=np.float32)
    half = int(img_size / 2.)

    if half_type == 0:  # top mask
        mask[:half, :] = 0.
    elif half_type == 1:  # bottom mask
        mask[half:, :] = 0.
    elif half_type == 2:  # left mask
        mask[:, :half] = 0.
    elif half_type == 3:  # right mask
        mask[:, half:] = 0.
    else:
        raise NotImplementedError

    return mask


def pattern_mask(img_size):
    num_points, ratio = 3, 0.25
    mask = np.zeros((img_size, img_size), dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for num in range(num_points):
        coordinate = np.random.randint(img_size, size=2)
        mask[coordinate[0], coordinate[1]] = 1.
        mask = cv2.dilate(mask, kernel, iterations=1)

    while np.sum(mask) < ratio * img_size * img_size:
        flag = True
        while flag:
            coordinate = np.random.randint(img_size, size=2)
            if mask[coordinate[0], coordinate[1]] == 1.:
                mask2 = np.zeros((img_size, img_size), dtype=np.float32)
                mask2[coordinate[0], coordinate[1]] = 1.
                mask2 = cv2.dilate(mask2, kernel, iterations=1)

                mask[mask + mask2 >= 1.] = 1.
                flag = False

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return 1. - mask

