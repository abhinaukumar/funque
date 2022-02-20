#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

import os
import sys
import re

import numpy as np
import pandas as pd
from funque.config import DisplayConfig

from funque.core.result_store import FileSystemResultStore
from funque.tools.misc import import_python_file, get_cmd_option, cmd_option_exists
from funque.core.quality_runner import FunqueQualityRunner
from funque.config import FunqueConfig
from funque.routine import run_test_on_dataset, print_matplotlib_warning
from funque.tools.stats import ListStats

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']

SUBJECTIVE_MODELS = ['DMOS', 'DMOS_MLE', 'MLE', 'MLE_CO_AP',
                     'MLE_CO_AP2 (default)', 'MOS', 'SR_DMOS',
                     'SR_MOS (i.e. ITU-R BT.500)',
                     'BR_SR_MOS (i.e. ITU-T P.913)',
                     'ZS_SR_DMOS', 'ZS_SR_MOS', '...']


def print_usage():
    print("usage: " + os.path.basename(sys.argv[0]) + \
          " test_dataset_list_filepath model_path " \
          "[--phone-model] [--subj-model subjective_model] [--cache-result] " \
          "[--parallelize] [--print-result] [--save-plot plot_dir] [--plot-wh plot_wh] "
          "[--processes processes] [--csv-suffix csv_suffix]\n")
    print("subjective_model:\n\t" + "\n\t".join(SUBJECTIVE_MODELS) + "\n")
    print("plot_wh: plot width and height in inches, example: 5x5 (default)")
    print("processes: must be an integer >=1")


def main():
    if len(sys.argv) < 3:
        print_usage()
        return 2

    csv_suffix = get_cmd_option(sys.argv, 3, len(sys.argv), '--csv-suffix')

    try:
        test_dataset_list_filepath = sys.argv[1]
    except ValueError:
        print_usage()
        return 2

    try:
        model_path = sys.argv[2]
    except ValueError:
        print_usage()
        return 2

    cache_result = cmd_option_exists(sys.argv, 3, len(sys.argv), '--cache-result')
    parallelize = cmd_option_exists(sys.argv, 3, len(sys.argv), '--parallelize')
    processes = get_cmd_option(sys.argv, 3, len(sys.argv), '--processes')
    print_result = cmd_option_exists(sys.argv, 3, len(sys.argv), '--print-result')
    suppress_plot = cmd_option_exists(sys.argv, 3, len(sys.argv), '--suppress-plot')
    phone_model = cmd_option_exists(sys.argv, 3, len(sys.argv), '--phone-model')

    pool_method = get_cmd_option(sys.argv, 3, len(sys.argv), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print('--pool can only have option among {}'.format(', '.join(POOL_METHODS)))
        return 2

    subj_model = get_cmd_option(sys.argv, 3, len(sys.argv), '--subj-model')

    try:
        from sureal.subjective_model import SubjectiveModel
        if subj_model is not None:
            subj_model_class = SubjectiveModel.find_subclass(subj_model)
        else:
            subj_model_class = SubjectiveModel.find_subclass('MLE_CO_AP2')
    except Exception as e:
        print("Error: " + str(e))
        return 1

    save_plot_dir = get_cmd_option(sys.argv, 3, len(sys.argv), '--save-plot')

    plot_wh = get_cmd_option(sys.argv, 3, len(sys.argv), '--plot-wh')
    if plot_wh is not None:
        try:
            mo = re.match(r"([0-9]+)x([0-9]+)", plot_wh)
            assert mo is not None
            w = mo.group(1)
            h = mo.group(2)
            w = int(w)
            h = int(h)
            plot_wh = (w, h)
        except Exception as e:
            print("Error: plot_wh must be in the format of WxH, example: 5x5")
            return 1

    runner_class = FunqueQualityRunner

    if processes is not None:
        try:
            processes = int(processes)
        except ValueError:
            print("Input error: processes must be an integer")
        assert processes >= 1

    try:
        test_dataset_list_store = import_python_file(test_dataset_list_filepath)
    except Exception as e:
        print("Error: " + str(e))
        return 1

    assert hasattr(test_dataset_list_store, 'datasets'), 'Test dataset list file must contain \'datasets\'.'
    assert type(test_dataset_list_store.datasets) in [list, tuple], '\'datasets\' attribute must be either a list or tuple,'
    test_dataset_files = test_dataset_list_store.datasets

    if cache_result:
        result_store = FileSystemResultStore(FunqueConfig.file_result_store_path())
    else:
        result_store = None

    # pooling
    if pool_method == 'harmonic_mean':
        aggregate_method = ListStats.harmonic_mean
    elif pool_method == 'min':
        aggregate_method = np.min
    elif pool_method == 'median':
        aggregate_method = np.median
    elif pool_method == 'perc5':
        aggregate_method = ListStats.perc5
    elif pool_method == 'perc10':
        aggregate_method = ListStats.perc10
    elif pool_method == 'perc20':
        aggregate_method = ListStats.perc20
    else: # None or 'mean'
        aggregate_method = np.mean

    if phone_model:
        enable_transform_score = True
    else:
        enable_transform_score = None

    srocc_dict = {}
    pcc_dict = {}
    rmse_dict = {}
    for test_dataset_file in test_dataset_files:
        try:
            test_dataset = import_python_file(test_dataset_file)
        except Exception as e:
            print("Error: " + str(e))
            return 1

        try:
            if suppress_plot:
                raise AssertionError

            from . import plt
            if plot_wh is None:
                plot_wh = (5, 5)
            _, ax = plt.subplots(figsize=plot_wh, nrows=1, ncols=1)

            _, results = run_test_on_dataset(test_dataset, runner_class, ax,
                                                result_store, model_path,
                                                parallelize=parallelize,
                                                aggregate_method=aggregate_method,
                                                subj_model_class=subj_model_class,
                                                enable_transform_score=enable_transform_score,
                                                processes=processes,
                                                return_stats=True
                                                )

            bbox = {'facecolor':'white', 'alpha':0.5, 'pad':20}
            ax.annotate('Testing Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

            # ax.set_xlim([-10, 110])
            # ax.set_ylim([-10, 110])

            plt.tight_layout()

            if save_plot_dir is None:
                DisplayConfig.show()
            else:
                DisplayConfig.show(write_to_dir=save_plot_dir)

        except ImportError:
            print_matplotlib_warning()
            _, results = run_test_on_dataset(test_dataset, runner_class, None,
                                                result_store, model_path,
                                                parallelize=parallelize,
                                                aggregate_method=aggregate_method,
                                                subj_model_class=subj_model_class,
                                                enable_transform_score=enable_transform_score,
                                                processes=processes,
                                                return_stats=True
                                                )
        except AssertionError:
            _, results = run_test_on_dataset(test_dataset, runner_class, None,
                                                result_store, model_path,
                                                parallelize=parallelize,
                                                aggregate_method=aggregate_method,
                                                subj_model_class=subj_model_class,
                                                enable_transform_score=enable_transform_score,
                                                processes=processes,
                                                return_stats=True
                                                )

        srocc_dict[test_dataset.dataset_name] = results['SRCC']
        pcc_dict[test_dataset.dataset_name] = results['PCC']
        rmse_dict[test_dataset.dataset_name] = results['RMSE']

    # Collect results into data frames
    df_pcc = pd.DataFrame()
    df_pcc = df_pcc.append(pcc_dict, ignore_index=True)
    df_srocc = pd.DataFrame()
    df_srocc = df_srocc.append(srocc_dict, ignore_index=True)
    df_rmse = pd.DataFrame()
    df_rmse = df_rmse.append(rmse_dict, ignore_index=True)

    def fisher_agg(x):
        z = np.mean(np.log(1 + x) - np.log(1 - x))
        t = np.exp(z)
        return (t - 1) / (t + 1)

    # Aggregate results across databases
    df_pcc['Average'] = df_pcc.apply(fisher_agg, axis=1)
    df_srocc['Average'] = df_srocc.apply(fisher_agg, axis=1)
    df_rmse['Average'] = df_rmse.apply(lambda row: np.mean(row), axis=1) # Beware: scores may not be scaled correctly

    # TODO: Find a better location to save results
    df_srocc.to_csv('srocc' + ('_' + csv_suffix if csv_suffix else '') + '.csv')
    df_pcc.to_csv('pcc' + ('_' + csv_suffix if csv_suffix else '') + '.csv')
    df_rmse.to_csv('rmse' + ('_' + csv_suffix if csv_suffix else '') + '.csv')
    if print_result:
        print('PCC')
        print(df_pcc)
        print('SROCC')
        print(df_srocc)
        print('RMSE')
        print(df_rmse)
    else:
        print('Avg. SROCC: {}, Avg. PCC: {}, Avg. RMSE: {}'.format(df_srocc['Average'].iloc[0], df_pcc['Average'].iloc[0], df_rmse['Average'].iloc[0]))
    print('Warning: run_multi_testing.py does not normalize subjective scores before computing performance stats. So, the average RMSE may be more senstive to some database(s) than others.')
    return


if __name__ == '__main__':
    ret = main()
    exit(ret)
