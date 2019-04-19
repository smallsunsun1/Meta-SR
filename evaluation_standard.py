import numpy as np
import cv2


def PSNR(target, ref, scale):
    target_data = np.asarray(target, dtype=np.float32)
    ref_data = np.asarray(ref, dtype=np.float32)
    target_y = cv2.cvtColor(target_data, cv2.COLOR_BGR2YCrCb)
    ref_y = cv2.cvtColor(ref_data, cv2.COLOR_BGR2YCrCb)
    diff = ref_y - target_y
    shave = scale
    diff = diff[shave:-shave, shave:-shave]
    mse = np.mean((diff / 255) ** 2)
    if mse == 0:
        return 100
    return -10 * np.log10(mse)