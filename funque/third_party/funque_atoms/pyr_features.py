from .dlm_utils import dlm_decouple, dlm_csf_filter, dlm_contrast_mask
from .vif_utils import vif_spatial, vif_channel_est, vif_gsm_model, im2col
import numpy as np


def dlm_pyr(pyr_ref, pyr_dist, border_size=0.2, full=False, csf='li'):
    assert len(pyr_ref) == len(pyr_dist), 'Pyramids must be of equal height.'

    pyr_ref.reverse()
    pyr_dist.reverse()

    pyr_rest, pyr_add = dlm_decouple(pyr_ref, pyr_dist)

    pyr_ref = dlm_csf_filter(pyr_ref, csf)
    pyr_rest = dlm_csf_filter(pyr_rest, csf)
    pyr_add = dlm_csf_filter(pyr_add, csf)

    pyr_rest, pyr_add = dlm_contrast_mask(pyr_rest, pyr_add)

    # flatten into a list of subbands for convenience
    pyr_ref = [item for sublist in pyr_ref for item in sublist]
    pyr_rest = [item for sublist in pyr_rest for item in sublist]
    pyr_add = [item for sublist in pyr_add for item in sublist]

    # pool results
    dlm_num = 0
    dlm_den = 0
    for subband in pyr_rest:
        h, w = subband.shape
        border_h = int(border_size*h)
        border_w = int(border_size*w)
        dlm_num += np.power(np.sum(np.power(subband[border_h:-border_h, border_w:-border_w], 3.0)), 1.0/3)
    for subband in pyr_ref:
        h, w = subband.shape
        border_h = int(border_size*h)
        border_w = int(border_size*w)
        dlm_den += np.power(np.sum(np.power(np.abs(subband[border_h:-border_h, border_w:-border_w]), 3.0)), 1.0/3)

    dlm_ret = (dlm_num + 1e-4) / (dlm_den + 1e-4)

    if full:
        aim_ret = 0
        count = 0
        for subband in pyr_add:
            h, w = subband.shape
            border_h = int(border_size*h)
            border_w = int(border_size*w)
            aim_ret += np.power(np.sum(np.power(subband[border_h:-border_h, border_w:-border_w], 3.0)), 1.0/3)
            count += (h - 2*border_h)*(w - 2*border_w)
        aim_ret /= count

        comb_ret = dlm_ret - 0.815 * (0.5 - 1.0 / (1.0 + np.exp(1375*aim_ret)))
        ret = (dlm_ret, aim_ret, comb_ret)
    else:
        ret = dlm_ret

    return ret


def vif_pyr(pyr_ref, pyr_dist, full=False, block_size=3, sigma_nsq=0.1):

    n_levels = len(pyr_ref)-1
    pyr_ref_ = {}
    pyr_dist_ = {}
    subband_keys = []
    for i in range(n_levels):
        for j in range(2):
            pyr_ref_[(n_levels-1-i, j)] = pyr_ref[i+1][j]
            pyr_dist_[(n_levels-1-i, j)] = pyr_dist[i+1][j]
            subband_keys.append((n_levels-1-i, j))
    pyr_ref_[n_levels] = pyr_ref[0]
    pyr_dist_[n_levels] = pyr_dist[0]
    subband_keys.append(n_levels)

    subband_keys.reverse()
    n_subbands = len(subband_keys)

    nums = np.zeros((n_subbands,))
    dens = np.zeros((n_subbands,))

    if block_size == 1:
        for i, subband in enumerate(subband_keys):
            nums[i], dens[i], _ = vif_spatial(pyr_ref_[subband], pyr_dist_[subband], k=9, sigma_nsq=sigma_nsq, full=True)
        if n_levels == 1:
            coeffs = [0.45, 0.45, 0.10]

            edge_ref = np.zeros_like(pyr_ref_[(0, 0)])
            for e, c in zip(pyr_ref[1], coeffs):
                edge_ref += c * e**2
            edge_ref = np.sqrt(edge_ref)

            edge_dist = np.zeros_like(pyr_dist_[(0, 0)])
            for e, c in zip(pyr_dist[1], coeffs):
                edge_dist += c * e**2
            edge_dist = np.sqrt(edge_dist)

            vif_edge = vif_spatial(edge_ref, edge_dist, k=9, sigma_nsq=sigma_nsq)
            vif_approx = (nums[0] + 1e-4) / (dens[0] + 1e-4)
            vif_dwt = 0.85*vif_approx + 0.15*vif_edge
            vif_val = np.mean(nums + 1e-4)/np.mean(dens + 1e-4)
            if not full:
                return vif_val
            else:
                return vif_val, vif_dwt, vif_edge, nums, dens
    else:
        M = block_size

        [g_all, sigma_vsq_all] = vif_channel_est(pyr_ref_, pyr_dist_, subband_keys, M)

        [s_all, lamda_all] = vif_gsm_model(pyr_ref_, subband_keys, M)

        for i in range(n_subbands):
            g = g_all[i]
            sigma_vsq = sigma_vsq_all[i]
            s = s_all[i]
            lamda = lamda_all[i]

            n_eigs = len(lamda)

            lev = subband_keys[i][0]+1 if isinstance(subband_keys[i], tuple) else subband_keys[i]

            winsize = 2**lev + 1
            offset = (winsize - 1)/2
            offset = int(np.ceil(offset/M))

            g = g[offset:-offset, offset:-offset]
            sigma_vsq = sigma_vsq[offset:-offset, offset:-offset]
            s = s[offset:-offset, offset:-offset]

            for j in range(n_eigs):
                nums[i] += np.mean(np.log(1 + g*g*s*lamda[j]/(sigma_vsq+sigma_nsq)))
                dens[i] += np.mean(np.log(1 + s*lamda[j]/sigma_nsq))

    vif_val = np.mean(nums + 1e-4)/np.mean(dens + 1e-4)
    if not full:
        return vif_val
    else:
        return vif_val, (nums + 1e-4), (dens + 1e-4)


def vif_dwt_pyr(pyr_ref, pyr_dist, k=3, stride=None, sigma_nsq=5, beta=0.94, full=False):
    if stride is None:
        stride = k

    levels = len(pyr_ref) - 1

    approx_ref = pyr_ref[0]
    approx_dist = pyr_dist[0]
    vif_approx = vif_spatial(approx_ref, approx_dist, k=k, sigma_nsq=sigma_nsq, stride=stride)

    subband_factors = [0.45, 0.45, 0.10]
    edge_ref = np.zeros(pyr_ref[1][0].shape)  # Infer size from a subband edge map
    edge_dist = np.zeros(pyr_dist[1][0].shape)  # Infer size from a subband edge map

    for i in range(levels):
        level_edge_ref = np.zeros_like(edge_ref)
        level_edge_dist = np.zeros_like(edge_dist)

        for subband, subband_edge_map in enumerate(pyr_ref[i+1]):
            level_edge_ref += subband_factors[subband] * subband_edge_map**2
        edge_ref += np.sqrt(level_edge_ref)

        for subband, subband_edge_map in enumerate(pyr_dist[i+1]):
            level_edge_dist += subband_factors[subband] * subband_edge_map**2
        edge_dist += np.sqrt(level_edge_dist)

    vif_edge = vif_spatial(edge_ref, edge_dist, k=k, sigma_nsq=sigma_nsq, stride=stride)

    if not full:
        return vif_approx * beta + vif_edge * (1 - beta)
    else:
        return vif_approx * beta + vif_edge * (1 - beta), vif_approx, vif_edge


def ssim_pyr(pyr_ref, pyr_dist, max_val=1, K1=0.01, K2=0.03, pool='cov'):
    assert len(pyr_ref) == len(pyr_dist), 'Both wavelet pyramids must be of the same height'
    n_levels = len(pyr_ref) - 1
    res_shape = pyr_ref[0].shape

    C1 = (K1*max_val)**2
    C2 = (K2*max_val)**2

    mu_x = np.zeros(res_shape)
    mu_y = np.zeros(res_shape)
    var_x = np.zeros(res_shape)
    var_y = np.zeros(res_shape)
    cov_xy = np.zeros(res_shape)

    win_dim = (1 << n_levels)  # 2^L
    win_size = (1 << (n_levels << 1))  # 2^(2L), i.e., a win_dim X win_dim square

    mu_x = pyr_ref[0] / win_dim
    mu_y = pyr_dist[0] / win_dim

    var_x = pyr_ref[1][0]**2
    var_y = pyr_dist[1][0]**2
    cov_xy = pyr_ref[1][0]*pyr_dist[1][0]
    for subband in range(1, 3):
        var_x += pyr_ref[1][subband]**2
        var_y += pyr_dist[1][subband]**2
        cov_xy += pyr_ref[1][subband]*pyr_dist[1][subband]

    block_size = 1
    for i in range(1, n_levels):
        block_size <<= 1
        for subband in range(3):
            var_x += im2col(pyr_ref[i+1][subband]**2, block_size, block_size).sum(0).reshape(var_x.shape)
            var_y += im2col(pyr_dist[i+1][subband]**2, block_size, block_size).sum(0).reshape(var_y.shape)
            cov_xy += im2col(pyr_ref[i+1][subband]*pyr_dist[i+1][subband], block_size, block_size).sum(0).reshape(var_y.shape)

    var_x /= win_size
    var_y /= win_size
    cov_xy /= win_size

    l = (2*mu_x*mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
    cs = (2 * cov_xy + C2) / (var_x + var_y + C2)

    ssim_map = l * cs
    mean_ssim = np.mean(ssim_map)

    if pool == 'mean':
        return mean_ssim
    elif pool == 'cov':
        return np.std(ssim_map) / mean_ssim
    elif pool == 'all':
        return mean_ssim, np.std(ssim_map) / mean_ssim
    else:
        raise ValueError('Invalid pool option.')
