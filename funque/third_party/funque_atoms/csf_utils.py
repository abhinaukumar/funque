import numpy as np


def csf_frequency(f):
    return (0.31 + 0.69*f) * np.exp(-0.29*f)


def csf_spat_filter(d2h, k=21):
    assert isinstance(k, int) and k > 0 and (k & 1), 'The length of the filter must be an odd positive integer'
    del_theta = 180 / (d2h * 1080 * np.pi)
    t = np.arange(-(k >> 1), (k >> 1) + 1) * del_theta
    assert len(t) == k, 'Filter is of the wrong size'

    a = 0.31
    b = 0.69
    c = 0.29
    f = 2*((a*c + b)*c**2 + (a*c - b)*4*np.pi**2 * t**2) / (c**2 + 4*np.pi**2 * t**2)**2   # Inverse Fourier Transform of CSF.

    return f*del_theta


# Ref "Most apparent distortion: full-reference image quality assessment and the role of strategy", E. C. Larson and D. M. Chandler
def csf_mannos_daly(f, theta):
    f_theta = f / (0.15*np.cos(theta) + 0.85)
    lamda = 0.228
    f_max = 4  # TODO: Find out how f_max is obtained from lamda
    if isinstance(f, np.ndarray):
        return np.where(f >= f_max, 2.6*(0.0192 + lamda*f_theta)*np.exp(-np.power(lamda*f_theta, 1.1)), 0.981)
    else:
        return 2.6*(0.0192 + lamda*f_theta)*np.exp(-np.power(lamda*f_theta, 1.1)) if f >= f_max else 0.981


def detection_threshold(a, k, f0, g, level_frequency):
    return a * np.power(10, k*np.log10(f0 * g / level_frequency)**2)


def csf_li(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    '''
    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))
    orientation_factors = [1.0/0.85, 1.0, 1.0, 1/(0.85-0.15)]
    lookup = np.array([[0.00150387263, 0.00544585178, 0.00544585178, 0.00023055401], 
                       [0.09476531388, 0.16683506215, 0.16683506215, 0.04074566701],
                       [0.54231822084, 0.66786346496, 0.66786346496, 0.38921962529],
                       [0.95194661972, 0.98626459244, 0.98626459244, 0.87735995465] ,
                       [0.95462489181, 0.91608864363, 0.91608864363, 0.98675189575]])
    if level < 5:
        return lookup[level, subband]
    else:
        return csf_frequency(level_frequency * orientation_factors[subband])


def csf_cdf97_watson(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: A. Watson, G. Yang, et al. "Visibility of Wavelet Quantization Noise"
    '''
    # Detection threshold model parameters
    a = 0.495
    k = 0.466
    f0 = 0.401
    gs = [1.501, 1, 1, 0.534]

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    # Basis function amplitudes
    amplitudes = np.array([[0.621710, 0.672340, 0.672340, 0.727090],
                           [0.345370, 0.413170, 0.413170, 0.494280],
                           [0.180040, 0.227270, 0.227270, 0.286880],
                           [0.091401, 0.117920, 0.117920, 0.152140],
                           [0.045943, 0.059758, 0.059758, 0.077727],
                           [0.023013, 0.030018, 0.030018, 0.039156]])

    return 0.5 * amplitudes[level, subband] / detection_threshold(a, k, f0, gs[subband], level_frequency)


def csf_dwt_hill(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: P. Hill, A. Achim, et al. "Contrast Sensitivity of the Wavelet, Dual Tree Complex Wavelet, Curvelet, and Steerable Pyramid Transforms"
    '''
    # Detection threshold model parameters
    a = 2.818
    k = 0.783
    f0 = 0.578
    gs = [1.5, 1, 1, 0.534]  # g0, i.e. for approximation subband, is not provided. Using value from Watson (ref: csf_cdf97_watson). Do not recommend using.

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    return 1.0 / (20 * np.log10(detection_threshold(a, k, f0, gs[subband], level_frequency) / 128))


def csf_dtcwt_hill(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: P. Hill, A. Achim, et al. "Contrast Sensitivity of the Wavelet, Dual Tree Complex Wavelet, Curvelet, and Steerable Pyramid Transforms"
    '''
    # Detection threshold model parameters
    a = 3.107
    k = 1.025
    f0 = 0.755
    gs = [1.3, 1, 1, 0.814]  # g0, i.e. for approximation subband, is not provided. Guessing. Do not use recommend using.

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    return 1.0 / (20 * np.log10(detection_threshold(a, k, f0, gs[subband], level_frequency) / 128))


def csf_curvelet_hill(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: P. Hill, A. Achim, et al. "Contrast Sensitivity of the Wavelet, Dual Tree Complex Wavelet, Curvelet, and Steerable Pyramid Transforms"
    '''
    # Detection threshold model parameters
    a = 1.083
    k = 0.790
    f0 = 0.509
    gs = [1, 1, 1, 1]  # g0, i.e. for approximation subband, is not provided. Guessing. Do not use recommend using.

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    return 1.0 / (20 * np.log10(detection_threshold(a, k, f0, gs[subband], level_frequency) / 128))


def csf_steerable_hill(level, subband):
    '''
    level: 0 indexed level of the discrete wavelet transform
    subband: 0 indexed in the list [approximation, horizontal, vertical, diagonal]
    Ref: P. Hill, A. Achim, et al. "Contrast Sensitivity of the Wavelet, Dual Tree Complex Wavelet, Curvelet, and Steerable Pyramid Transforms"
    '''
    # Detection threshold model parameters
    a = 2.617
    k = 0.960
    f0 = 0.487
    gs = [1, 1, 1, 1]  # g0, i.e. for approximation subband, is not provided. Guessing. Do not use recommend using.

    # Distance to height ratio of the display
    d2h = 3.0
    pic_height = 1080
    factor = np.pi*pic_height*d2h/180
    level_frequency = factor / (1 << (level+1))

    return 1.0 / (20 * np.log10(detection_threshold(a, k, f0, gs[subband], level_frequency) / 128))


def ahc_weight(level, subband, n_levels, binarized=True):
    '''
    Weighting function used in Adaptive High Frequency Clipping
    If binarized is True, weight is compared to a threshold and 0-1 outputs are returned.
    Ref: K. Gu, G. Zhai, et al. "Adaptive High Frequency Clipping for Improved Image Quality Assessment"
    '''
    # Weighting function parameters
    a = 10
    k = 10
    t = 2
    d0 = 512
    gs = [1, 2, 2, 1]  # g0, i.e. for approximation subband, is not provided. Guessing. Do not use recommend using.
    thresh = 1.0

    # Distance to height ratio of the display
    d2h = 3.0

    weight = gs[subband] * np.power(k, t*(n_levels - (level+1))) / np.power(a,  d2h/d0)  # Paper says "d/d0" but I think they meant to use d2h
    if binarized:
        weight = float(weight >= thresh)

    return weight


csf_dict = {'frequency': csf_frequency,
            'spat_filter': csf_spat_filter,
            'spat_filter_clipped': csf_spat_filter,
            'li': csf_li,
            'mannos_daly': csf_mannos_daly,
            'cdf97_watson': csf_cdf97_watson,
            'dwt_hill': csf_dwt_hill,
            'dtcwt_hill': csf_dtcwt_hill,
            'curvelet_hill': csf_curvelet_hill,
            'steerable_hill': csf_steerable_hill,
            'ahc': ahc_weight}
