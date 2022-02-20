#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

import os
import sys
import re

from itertools import chain, combinations

import numpy as np
import pandas as pd
from funque.config import DisplayConfig

from funque.core.result_store import FileSystemResultStore
from funque.tools.misc import import_python_file, get_cmd_option, cmd_option_exists
from funque.core.quality_runner import FunqueQualityRunner
from funque.config import FunqueConfig
from funque.routine import run_test_on_dataset,  random_cv_on_dataset, train_test_on_dataset, print_matplotlib_warning
from funque.tools.stats import ListStats

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']

SUBJECTIVE_MODELS = ['DMOS', 'DMOS_MLE', 'MLE', 'MLE_CO_AP',
                     'MLE_CO_AP2 (default)', 'MOS', 'SR_DMOS',
                     'SR_MOS (i.e. ITU-R BT.500)',
                     'BR_SR_MOS (i.e. ITU-T P.913)',
                     'ZS_SR_DMOS', 'ZS_SR_MOS', '...']


def get_feature_param(feature_list):
    feature_dict = {}
    for key, value in feature_list:
        feature_dict.setdefault(key, []).append(value)
    feature_param = lambda:0
    feature_param.feature_dict = feature_dict
    return feature_param


def combinationset(iterable):
    s = list(iterable)
    sizes = np.array([len(e) for e in s])
    sizes = sizes[::-1]
    tot = np.prod(sizes + 1)
    ret = []
    for i in range(1, tot):
        num = i
        temp = []
        for size in sizes:
            temp.append(num % (size+1))
            num //= (size + 1)
        temp = temp[::-1]
        ret.append([s[set_ind][obj_ind-1] for set_ind, obj_ind in enumerate(temp) if obj_ind > 0])
    return ret


# Power set routine from itertools docs
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def print_usage():
    print("usage: " + os.path.basename(sys.argv[0]) + \
          " feature_param_filepath train_dataset_filepath test_dataset_list_filepath model_param_filepath [--config-param config_param_filepath] " \
          "[--phone-model] [--subj-model subjective_model] [--cache-result] " \
          "[--parallelize] [--print-result] [--save-plot plot_dir] [--plot-wh plot_wh] [--csv-suffix csv_suffix] "
          "[--processes processes]\n")
    print("subjective_model:\n\t" + "\n\t".join(SUBJECTIVE_MODELS) + "\n")
    print("plot_wh: plot width and height in inches, example: 5x5 (default)")
    print("processes: must be an integer >=1")


def train_model(feature_param, args, cv=False):

    try:
        train_dataset_filepath = args[2]
        model_param_filepath = args[4]
    except ValueError:
        print_usage()
        return 2

    output_model_filepath = os.path.join(FunqueConfig.model_path(), 'multi_test_temp_model.pkl')

    try:
        train_dataset = import_python_file(train_dataset_filepath)
        model_param = import_python_file(model_param_filepath)
    except Exception as e:
        print("Error: %s" % e)
        return 1

    cache_result = cmd_option_exists(args, 5, len(args), '--cache-result')
    parallelize = cmd_option_exists(args, 5, len(args), '--parallelize')
    processes = get_cmd_option(args, 5, len(args), '--processes')
    suppress_plot = cmd_option_exists(args, 5, len(args), '--suppress-plot')

    splits = get_cmd_option(args, 5, len(args), '--splits')
    if splits is None:
        splits = 5000
    else:
        splits = int(splits)
        assert splits > 0, 'splits must be a positive integer'

    pool_method = get_cmd_option(args, 5, len(args), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print('--pool can only have option among {}'.format(', '.join(POOL_METHODS)))
        return 2

    subj_model = get_cmd_option(args, 5, len(args), '--subj-model')

    try:
        from sureal.subjective_model import SubjectiveModel
        if subj_model is not None:
            subj_model_class = SubjectiveModel.find_subclass(subj_model)
        else:
            subj_model_class = SubjectiveModel.find_subclass('MLE_CO_AP2')
    except Exception as e:
        print("Error: " + str(e))
        return 1

    save_plot_dir = get_cmd_option(args, 5, len(args), '--save-plot')

    plot_wh = get_cmd_option(args, 5, len(args), '--plot-wh')
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

    if processes is not None:
        try:
            processes = int(processes)
        except ValueError:
            print("Input error: processes must be an integer")
        assert processes >= 1

    config_param_filepath = get_cmd_option(sys.argv, 3, len(sys.argv), '--config-param-filepath')
    if config_param_filepath is not None:
        config_param = import_python_file(config_param_filepath)
        optional_dict = config_param.optional_dict
    else:
        optional_dict = None

    try:
        train_dataset = import_python_file(train_dataset_filepath)
    except Exception as e:
        print("Error: " + str(e))
        return 1

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

    logger = None

    try:
        if suppress_plot:
            raise AssertionError

        from funque import plt
        _, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
        axs = (ax, None)

        if not cv:
            _, _, stats, _, _, _, _ = train_test_on_dataset(train_dataset=train_dataset, test_dataset=None,
                                                                 feature_param=feature_param, model_param=model_param,
                                                                 train_ax=axs[0], test_ax=axs[1],
                                                                 result_store=result_store,
                                                                 parallelize=parallelize,
                                                                 logger=logger,
                                                                 output_model_filepath=output_model_filepath,
                                                                 aggregate_method=aggregate_method,
                                                                 subj_model_class=subj_model_class,
                                                                 processes=processes,
                                                                 optional_dict=optional_dict,
                                                                 )
        else:
            _, cv_output = random_cv_on_dataset(dataset=train_dataset,
                                                feature_param=feature_param, model_param=model_param,
                                                ax=axs[0],
                                                result_store=result_store,
                                                parallelize=parallelize,
                                                logger=logger,
                                                aggregate_method=aggregate_method,
                                                subj_model_class=subj_model_class,
                                                processes=processes,
                                                optional_dict=optional_dict,
                                                splits=splits
                                                )
            stats = cv_output['aggr_stats']

        bbox = {'facecolor':'white', 'alpha':0.5, 'pad':20}
        axs[0].annotate('Training Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)
        if axs[1] is not None:
            axs[1].annotate('Testing Set', xy=(0.1, 0.85), xycoords='axes fraction', bbox=bbox)

        # ax.set_xlim([-10, 110])
        # ax.set_ylim([-10, 110])

        plt.tight_layout()

        if save_plot_dir is None:
            DisplayConfig.show()
        else:
            DisplayConfig.show(write_to_dir=save_plot_dir)

    except ImportError:
        print_matplotlib_warning()
        if not cv:
            _, _, stats, _, _, _, _ = train_test_on_dataset(train_dataset=train_dataset, test_dataset=None,
                                                                 feature_param=feature_param, model_param=model_param,
                                                                 train_ax=None, test_ax=None,
                                                                 result_store=result_store,
                                                                 parallelize=parallelize,
                                                                 logger=logger,
                                                                 output_model_filepath=output_model_filepath,
                                                                 aggregate_method=aggregate_method,
                                                                 subj_model_class=subj_model_class,
                                                                 processes=processes,
                                                                 optional_dict=optional_dict,
                                                                 )
        else:
            _, cv_output = random_cv_on_dataset(dataset=train_dataset,
                                                feature_param=feature_param, model_param=model_param,
                                                ax=None,
                                                result_store=result_store,
                                                parallelize=parallelize,
                                                logger=logger,
                                                aggregate_method=aggregate_method,
                                                subj_model_class=subj_model_class,
                                                processes=processes,
                                                optional_dict=optional_dict,
                                                splits=splits
                                                )
            stats = cv_output['aggr_stats']

    except AssertionError:
        if not cv:
            _, _, stats, _, _, _, _ = train_test_on_dataset(train_dataset=train_dataset, test_dataset=None,
                                                                 feature_param=feature_param, model_param=model_param,
                                                                 train_ax=None, test_ax=None,
                                                                 result_store=result_store,
                                                                 parallelize=parallelize,
                                                                 logger=logger,
                                                                 output_model_filepath=output_model_filepath,
                                                                 aggregate_method=aggregate_method,
                                                                 subj_model_class=subj_model_class,
                                                                 processes=processes,
                                                                 optional_dict=optional_dict,
                                                                 )
        else:
            _, cv_output = random_cv_on_dataset(dataset=train_dataset,
                                                feature_param=feature_param, model_param=model_param,
                                                ax=None,
                                                result_store=result_store,
                                                parallelize=parallelize,
                                                logger=logger,
                                                aggregate_method=aggregate_method,
                                                subj_model_class=subj_model_class,
                                                processes=processes,
                                                optional_dict=optional_dict,
                                                splits=splits
                                                )
            stats = cv_output['aggr_stats']

    return output_model_filepath, stats


def multi_test_model(model_path, args):

    try:
        test_dataset_list_filepath = args[3]
    except ValueError:
        print_usage()
        return 2

    cache_result = cmd_option_exists(args, 5, len(args), '--cache-result')
    parallelize = cmd_option_exists(args, 5, len(args), '--parallelize')
    processes = get_cmd_option(args, 5, len(args), '--processes')
    suppress_plot = cmd_option_exists(args, 5, len(args), '--suppress-plot')
    phone_model = cmd_option_exists(args, 5, len(args), '--phone-model')

    pool_method = get_cmd_option(args, 5, len(args), '--pool')
    if not (pool_method is None
            or pool_method in POOL_METHODS):
        print('--pool can only have option among {}'.format(', '.join(POOL_METHODS)))
        return 2

    subj_model = get_cmd_option(args, 5, len(args), '--subj-model')

    try:
        from sureal.subjective_model import SubjectiveModel
        if subj_model is not None:
            subj_model_class = SubjectiveModel.find_subclass(subj_model)
        else:
            subj_model_class = SubjectiveModel.find_subclass('MLE_CO_AP2')
    except Exception as e:
        print("Error: " + str(e))
        return 1

    save_plot_dir = get_cmd_option(args, 5, len(args), '--save-plot')

    plot_wh = get_cmd_option(args, 5, len(args), '--plot-wh')
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

            from funque import plt
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

    return srocc_dict, pcc_dict, rmse_dict


def main():
    if len(sys.argv) < 5:
        print_usage()
        return 2

    csv_suffix = get_cmd_option(sys.argv, 5, len(sys.argv), '--csv-suffix')
    print_result = cmd_option_exists(sys.argv, 5, len(sys.argv), '--print-result')

    feature_param_filepath = sys.argv[1]
    feature_dict_store = import_python_file(feature_param_filepath)
    assert hasattr(feature_dict_store, 'feature_sections'), 'Feature params file must contain \'feature_sections\''
    full_feature_list = feature_dict_store.feature_sections
    common_feature_list = feature_dict_store.common_features if hasattr(feature_dict_store, 'common_features') else []

    cross_validate = cmd_option_exists(sys.argv, 5, len(sys.argv), '--cross-validate')

    full_feature_powerset = combinationset(full_feature_list)
    # If common features have been passed, you are allowed to omit all optional features.
    if len(common_feature_list) > 0:
        full_feature_powerset.append([])
    # Update feature list to include common features.
    full_feature_powerset = [common_feature_list + feature_list for feature_list in full_feature_powerset]

    df_srocc = pd.DataFrame(columns=['Features'])
    df_pcc = pd.DataFrame(columns=['Features'])
    df_rmse = pd.DataFrame(columns=['Features'])

    for feature_list in full_feature_powerset:
        feature_param = get_feature_param(feature_list)
        model_path, train_stats = train_model(feature_param, sys.argv, cv=cross_validate)
        if cross_validate:
            model_path, _ = train_model(feature_param, sys.argv, cv=False)

        srocc_dict, pcc_dict, rmse_dict = multi_test_model(model_path, sys.argv)
        srocc_dict['Training'] = train_stats['SRCC']
        pcc_dict['Training'] = train_stats['PCC']
        rmse_dict['Training'] = train_stats['RMSE']

        srocc_dict['Features'] = pcc_dict['Features'] = rmse_dict['Features'] = ' + '.join([e[0] + ': ' + e[1] for e in feature_list])
        df_srocc = df_srocc.append(srocc_dict, ignore_index=True)
        df_pcc = df_pcc.append(pcc_dict, ignore_index=True)
        df_rmse = df_rmse.append(rmse_dict, ignore_index=True)

    def fisher_agg(x):
        z = np.mean(np.log(1 + x) - np.log(1 - x))
        t = np.exp(z)
        return (t - 1) / (t + 1)

    scores_df = df_srocc.loc[:, df_srocc.columns.difference(['Features', 'Training'])]
    df_srocc['Average'] = scores_df.apply(fisher_agg, axis=1)

    scores_df = df_pcc.loc[:, df_pcc.columns.difference(['Features', 'Training'])]
    df_pcc['Average'] = scores_df.apply(fisher_agg, axis=1)

    scores_df = df_rmse.loc[:, df_rmse.columns.difference(['Features', 'Training'])]
    df_rmse['Average'] = scores_df.apply(lambda row: np.mean(row), axis=1) # Caution: scores may not be scaled correctly

    # df_srocc.sort_values('Average', ascending=False, inplace=True)
    # df_pcc.sort_values('Average', ascending=False, inplace=True)
    # df_rmse.sort_values('Average', ascending=False, inplace=True)

    df_srocc.sort_values('Training', ascending=False, inplace=True)
    df_pcc.sort_values('Training', ascending=False, inplace=True)
    df_rmse.sort_values('Training', ascending=False, inplace=True)

    # TODO: Find a better location to save results
    df_srocc.to_csv('srocc' + ('_' + csv_suffix if csv_suffix else '') + '.csv')
    df_pcc.to_csv('pcc' + ('_' + csv_suffix if csv_suffix else '') + '.csv')
    df_rmse.to_csv('rmse' + ('_' + csv_suffix if csv_suffix else '') + '.csv')

    if print_result:
        print(df_srocc.head())
        print(df_pcc.head())
        print(df_rmse.head())

    print('Warning: run_multi_testing.py does not normalize subjective scores before computing performance stats. So, the average RMSE may be more senstive to some database(s) than others.')
    return


if __name__ == '__main__':
    ret = main()
    exit(ret)
