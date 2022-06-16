import ast

import numpy as np
import scipy as sp

import cv2
from pywt import wavedec2, dwt2

from ..third_party.funque_atoms import pyr_features
from ..third_party.vmaf_atoms import vmaf_features
from ..core.feature_extractor import FeatureExtractor, VmafexecFeatureExtractorMixin

from ..tools.reader import YuvReader

from ..third_party.funque_atoms.filter_utils import filter_img


class PyVmafFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):
    '''
    A feature extractor that reimplements the VMAF feature set in Python.
    '''
    TYPE = 'PyVMAF_feature'
    VERSION = '1.0'

    ATOM_FEATURES = [
                    'vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3',  # VIF features
                    'dlm',  # DLM features (called ADM in VMAF)
                    'motion'  # Miscellaneous features, from VMAF.
                    ]
    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = dict(zip(ATOM_FEATURES, ATOM_FEATURES))

    MSVIF_LEVELS = 4

    def __init__(self, assets, **kwargs):
        super(FeatureExtractor, self).__init__(assets, **kwargs)

        self.vif_filters = [
            np.array([ 0.00745626912, 0.0142655009, 0.0250313189, 0.0402820669, 0.0594526194, 0.0804751068, 0.0999041125, 0.113746084, 0.118773937, 0.113746084, 0.0999041125, 0.0804751068, 0.0594526194, 0.0402820669, 0.0250313189, 0.0142655009, 0.00745626912 ]),
            np.array([ 0.0189780835, 0.0558981746, 0.120920904, 0.192116052, 0.224173605, 0.192116052, 0.120920904, 0.0558981746, 0.0189780835 ]),
            np.array([ 0.054488685, 0.244201347, 0.402619958, 0.244201347, 0.054488685 ]),
            np.array([ 0.166378498, 0.667243004, 0.166378498 ]),
        ]

    @staticmethod
    def _assert_bit_depth(ref_yuv_reader, dis_yuv_reader):
        if ref_yuv_reader._is_8bit():
            assert dis_yuv_reader._is_8bit()
        elif ref_yuv_reader._is_10bitle():
            assert dis_yuv_reader._is_10bitle()
        elif ref_yuv_reader._is_12bitle():
            assert dis_yuv_reader._is_12bitle()
        elif ref_yuv_reader._is_16bitle():
            assert dis_yuv_reader._is_16bitle()
        else:
            assert False, 'unknown bit depth and type'

    def _get_max_val(self, ref_yuv_reader):
        if self.optional_dict is not None and 'max_val' in self.optional_dict:
            assert type(self.optional_dict['max_val']) == int or float
            return self.optional_dict['max_val']
        elif ref_yuv_reader._is_8bit():
            return 255
        elif ref_yuv_reader._is_10bitle():
            return 1024
        elif ref_yuv_reader._is_12bitle():
            return 4095
        elif ref_yuv_reader._is_16bitle():
            return 65535
        else:
            assert False, 'unknown bit depth and type'

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate feature
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type = self._get_workfile_yuv_type(asset)
        w = quality_width
        h = quality_height

        log_dicts = list()
        with YuvReader(filepath=asset.ref_procfile_path, width=w, height=h,
                       yuv_type=yuv_type) as ref_yuv_reader:
            with YuvReader(filepath=asset.dis_procfile_path, width=w, height=h,
                           yuv_type=yuv_type) as dis_yuv_reader:

                self._assert_bit_depth(ref_yuv_reader, dis_yuv_reader)
                max_val = self._get_max_val(ref_yuv_reader)

                frm = 0
                while True:
                    try:
                        yuv_ref = ref_yuv_reader.next(format='float')
                        yuv_dis = dis_yuv_reader.next(format='float')
                    except StopIteration:
                        break

                    y_ref, _, _ = yuv_ref
                    y_dis, _, _ = yuv_dis

                    y_ref = y_ref / max_val
                    y_dis = y_dis / max_val

                    log_entry = {}
                    log_entry['frame'] = frm

                    # VIF features
                    vif_scale_ref = y_ref.copy()
                    vif_scale_dis = y_dis.copy()
                    for scale in range(self.MSVIF_LEVELS):
                        # Compute VIF at current scale
                        log_entry['vif_scale' + str(scale)] = vmaf_features.vif(vif_scale_ref, vif_scale_dis, self.vif_filters[scale])
                        # Filter and decimate
                        if scale != self.MSVIF_LEVELS-1:
                            vif_scale_ref = sp.ndimage.convolve1d(sp.ndimage.convolve1d(vif_scale_ref, self.vif_filters[scale], 0), self.vif_filters[scale], 1)
                            vif_scale_dis = sp.ndimage.convolve1d(sp.ndimage.convolve1d(vif_scale_dis, self.vif_filters[scale], 0), self.vif_filters[scale], 1)
                            vif_scale_ref = vif_scale_ref[::2, ::2]
                            vif_scale_dis = vif_scale_dis[::2, ::2]

                    dlm_val = vmaf_features.dlm(y_ref, y_dis)
                    log_entry['dlm'] = dlm_val

                    motion_val = vmaf_features.motion(y_ref, y_dis, self.vif_filters[2])
                    log_entry['motion'] = motion_val

                    log_dicts.append(log_entry)

                    frm += 1

        feature_result = {}
        for frm, log_dict in enumerate(log_dicts):
            assert frm == log_dict['frame']
            for feat in log_dict:
                feature_result.setdefault(self.get_scores_key(feat), []).append(log_dict[feat])

        # Write to logfile
        with open(log_file_path, 'wt') as log_file:
            log_file.write(str(feature_result))

    def _get_feature_scores(self, asset):
        log_file_path = self._get_log_file_path(asset)

        with open(log_file_path, 'rt') as log_file:
            log_str = log_file.read()
            feature_result = ast.literal_eval(log_str)

        return feature_result


class FunqueFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):
    '''
    A feature extractor that implements the proposed FUNQUE model.
    '''
    TYPE = 'FUNQUE_feature'
    VERSION = '1.0'

    ATOM_FEATURES = ['wd_essim',  # SSIM features
                     'vif_scale1', 'vif_scale2'  # VIF features using a 3x3 vector GSM model
                     'dlm',  # DLM features
                     'motion'  # Motion computed on approximation subband
                     ]

    def __init__(self, assets, **kwargs):
        super(FeatureExtractor, self).__init__(assets, **kwargs)
        self.wavelet = 'haar'
        self.wavelet_levels = 1
        self.vif_extra_levels = 1

    @staticmethod
    def _assert_bit_depth(ref_yuv_reader, dis_yuv_reader):
        if ref_yuv_reader._is_8bit():
            assert dis_yuv_reader._is_8bit()
        elif ref_yuv_reader._is_10bitle():
            assert dis_yuv_reader._is_10bitle()
        elif ref_yuv_reader._is_12bitle():
            assert dis_yuv_reader._is_12bitle()
        elif ref_yuv_reader._is_16bitle():
            assert dis_yuv_reader._is_16bitle()
        else:
            assert False, 'unknown bit depth and type'

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate feature
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type = self._get_workfile_yuv_type(asset)
        w = quality_width
        h = quality_height

        w_crop = ((w >> 2) << 2)
        h_crop = ((h >> 2) << 2)

        log_dicts = list()
        with YuvReader(filepath=asset.ref_procfile_path, width=w, height=h,
                       yuv_type=yuv_type) as ref_yuv_reader:
            with YuvReader(filepath=asset.dis_procfile_path, width=w, height=h,
                           yuv_type=yuv_type) as dis_yuv_reader:

                self._assert_bit_depth(ref_yuv_reader, dis_yuv_reader)

                frm = 0
                prev_approx = None
                while True:
                    try:
                        yuv_ref = ref_yuv_reader.next(format='uint')
                        yuv_dis = dis_yuv_reader.next(format='uint')

                        y_ref, _, _ = yuv_ref
                        y_dis, _, _ = yuv_dis

                        # Apply SAST.
                        if asset.enable_resizer:
                            y_ref = cv2.resize(y_ref, (w//2, h//2), interpolation=cv2.INTER_CUBIC)
                            y_dis = cv2.resize(y_dis, (w//2, h//2), interpolation=cv2.INTER_CUBIC)

                        if ref_yuv_reader._is_8bit():
                            y_ref = y_ref.astype(np.double) / (2.0**8 - 1.0)
                            y_dis = y_dis.astype(np.double) / (2.0**8 - 1.0)
                        elif ref_yuv_reader._is_10bitle():
                            y_ref = y_ref.astype(np.double) / (2.0**10 - 1.0)
                            y_dis = y_dis.astype(np.double) / (2.0**10 - 1.0)
                        elif ref_yuv_reader._is_12bitle():
                            y_ref = y_ref.astype(np.double) / (2.0**12 - 1.0)
                            y_dis = y_dis.astype(np.double) / (2.0**12 - 1.0)
                        elif ref_yuv_reader._is_16bitle():
                            y_ref = y_ref.astype(np.double) / (2.0**16 - 1.0)
                            y_dis = y_dis.astype(np.double) / (2.0**16 - 1.0)
                        else:
                            assert False

                    except StopIteration:
                        break

                    # Cropping to a power of 2 to avoid problems in WD-SSIM
                    y_ref = y_ref[:h_crop, :w_crop]
                    y_dis = y_dis[:h_crop, :w_crop]

                    log_entry = {}
                    log_entry['frame'] = frm

                    y_ref = filter_img(y_ref, 'spat_filter', wavelet=None, k=21)
                    y_dis = filter_img(y_dis, 'spat_filter', wavelet=None, k=21)

                    pyr_ref = wavedec2(y_ref, self.wavelet, 'periodization', self.wavelet_levels)
                    pyr_dis = wavedec2(y_dis, self.wavelet, 'periodization', self.wavelet_levels)

                    approxs_ref = [pyr_ref[0]]
                    approxs_dis = [pyr_dis[0]]

                    for _ in range(self.vif_extra_levels):
                        approxs_ref.append(dwt2(approxs_ref[0], self.wavelet, 'periodization')[0])
                        approxs_dis.append(dwt2(approxs_dis[0], self.wavelet, 'periodization')[0])

                    # WD-ESSIM feature
                    essim_val = pyr_features.ssim_pyr(pyr_ref, pyr_dis, pool='cov')
                    log_entry['wd_essim'] = essim_val

                    # VIF features
                    for scale, (approx_ref, approx_dis) in enumerate(zip(approxs_ref, approxs_dis)):
                        log_entry['vif_scale' + str(scale+1)] = pyr_features.vif_spatial(approx_ref, approx_dis, k=9, sigma_nsq=5.0, full=False)

                    # DLM feature
                    dlm_val = pyr_features.dlm_pyr(pyr_ref[1:], pyr_dis[1:], csf=None)
                    log_entry['dlm'] = dlm_val

                    # Motion feature
                    # Compute MAD between adjacent frames using the approximation subband as a blurred version of the frames.
                    motion_val = np.mean(np.abs(pyr_ref[0] - prev_approx)) if frm != 0 else 0
                    log_entry['motion'] = motion_val
                    prev_approx = pyr_ref[0]

                    log_dicts.append(log_entry)

                    frm += 1

        feature_result = {}
        for frm, log_dict in enumerate(log_dicts):
            assert frm == log_dict['frame']
            for feat in log_dict:
                feature_result.setdefault(self.get_scores_key(feat), []).append(log_dict[feat])

        # Write to logfile
        with open(log_file_path, 'wt') as log_file:
            log_file.write(str(feature_result))

    def _get_feature_scores(self, asset):
        log_file_path = self._get_log_file_path(asset)

        with open(log_file_path, 'rt') as log_file:
            log_str = log_file.read()
            feature_result = ast.literal_eval(log_str)

        return feature_result
