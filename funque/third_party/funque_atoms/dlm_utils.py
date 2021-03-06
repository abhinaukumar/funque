import numpy as np
from .csf_utils import csf_dict


def integral_image(x):
    M, N = x.shape
    int_x = np.zeros((M+1, N+1))
    int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
    return int_x


def integral_image_sums(x, k, stride=1):
    x_pad = np.pad(x, int((k - stride)/2), mode='reflect')
    int_x = integral_image(x_pad)
    ret = (int_x[:-k:stride, :-k:stride] - int_x[:-k:stride, k::stride] - int_x[k::stride, :-k:stride] + int_x[k::stride, k::stride])
    return ret


def csf(f):
    return (0.31 + 0.69*f) * np.exp(-0.29*f)


def dlm_decouple(pyr_ref, pyr_dist):
    eps = 1e-30
    n_levels = len(pyr_ref)
    pyr_rest = []
    pyr_add = []

    for level in range(n_levels):
        psi_ref = np.arctan(pyr_ref[level][1] / (pyr_ref[level][0] + eps)) + np.pi*(pyr_ref[level][0] <= 0)
        psi_dist = np.arctan(pyr_dist[level][1] / (pyr_dist[level][0] + eps)) + np.pi*(pyr_dist[level][0] <= 0)

        psi_diff = 180*np.abs(psi_ref - psi_dist)/np.pi
        mask = (psi_diff < 1)
        level_rest = []
        for i in range(3):
            k = np.clip(pyr_dist[level][i] / (pyr_ref[level][i] + eps), 0.0, 1.0)
            level_rest.append(k * pyr_ref[level][i])
            level_rest[i][mask] = pyr_dist[level][i][mask]

        pyr_rest.append(tuple(level_rest))

    for level_dist, level_rest in zip(pyr_dist, pyr_rest):
        level_add = []
        for i in range(3):
            level_add.append(level_dist[i] - level_rest[i])
        pyr_add.append(tuple(level_add))

    return pyr_rest, pyr_add


def dlm_csf_filter(pyr, csf):
    if csf is None:
        return pyr

    csf_funct = csf_dict[csf]
    n_levels = len(pyr)
    filt_pyr = []
    for level in range(n_levels):
        filt_level = []
        for subband in range(3):
            if csf != 'ahc':
                filt_level.append(pyr[level][subband] * csf_funct(level, subband+1))  # No approximation coefficient. Only H, V, D.
            else:
                filt_level.append(pyr[level][subband] * csf_funct(level, subband+1, n_levels))  # No approximation coefficient. Only H, V, D.
        filt_pyr.append(tuple(filt_level))

    return filt_pyr


# Masks pyr_1 using pyr_2
def dlm_contrast_mask_one_way(pyr_1, pyr_2):
    n_levels = len(pyr_1)
    masked_pyr = []
    for level in range(n_levels):
        masking_threshold = 0
        for i in range(3):
            masking_signal = np.abs(pyr_2[level][i])
            masking_threshold += (integral_image_sums(masking_signal, 3) + masking_signal) / 30
        masked_level = []
        for i in range(3):
            masked_level.append(np.clip(np.abs(pyr_1[level][i]) - masking_threshold, 0, None))
        masked_pyr.append(tuple(masked_level))
    return masked_pyr


# Masks each pyramid using the other
def dlm_contrast_mask(pyr_1, pyr_2):
    masked_pyr_1 = dlm_contrast_mask_one_way(pyr_1, pyr_2)
    masked_pyr_2 = dlm_contrast_mask_one_way(pyr_2, pyr_1)
    return masked_pyr_1, masked_pyr_2
