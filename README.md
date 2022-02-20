# FUNQUE
The official Python implementation of the FUNQUE framework for Video Quality Assessment (VQA), which provides both a significant improvement in performance compared to VMAF, couple with a drastic reduction in computational complexity.
In addition, we also provide the pretrained FUNQUE model for use off-the-shelf, under `models/`. A major part of this code has been forked from the Python library in Netflix's VMAF repository [1]. A summary of FUNQUE is below.
For more details, refer to "FUNQUE: Fusion of Unified Quality Evaluators".

## Features of FUNQUE
1. A unified transform used by all "atom" quality models that accounts for the human visual system (HVS), using spatial-domain contrast-sensitive filtering (CSF).
2. Using the Self-Adaptive Scale Transform (SAST) [2] to rescale videos prior to quality assessement.
3. Computing Enhanced SSIM [3, 4] directly from wavelet coefficients.

## Performance of FUNQUE
FUNQUE significantly outperforms VMAF (retrained on public data for a fair comparison) and also outperforms the high-complexity Enhanced VMAF - M2 [5] model that was trained on public data.
FUNQUE also rivals the performance of the other Enhanced VMAF models that use private Netflix data for training. The effect of the private data is seen in the difference between the performance of VMAF v0.6.1 and its retrained version.

| Model | Test Performance |
| ---------- | ------- |
|VMAF v0.6.1 |  0.8631 |
|__Retrained VMAF v0.6.1__ | __0.8019__ |
|Enhanced VMAF - M1 | 0.8761 |
|__Enhanced VMAF - M2__ | __0.8600__ |
|Enhanced VMAF | 0.8842 |
|__FUNQUE__ | __0.8715__ |

FUNQUE also runs in less than 1/8th of the time as compared to an equivalent Python implementation of VMAF (PyVMAF). The pretrained PyVMAF model is also available under `models`. Enhanced VMAF is more computationally expensive than VMAF, so it has not been included in this comparison.
| Model |	Running time (s) |	Ops Per Pixel |	Observed Speedup |	Expected Speedup |
| ----- | ------- | ------------- | ---------------- | ----------------- |
| PyVMAF	| 105.23 |	219.61 | 1 | 1 |
| __FUNQUE__ |	__12.73__ |	__39.30__ |	__8.265__ |	__5.588__ |

## Code Reference
1. To test pretrained models on a set of databases, list the database files in `resource/dataset/multi_test_datasets.py` and run
```
./test_models.sh <model> <number of processes>
```
where `model` is either `funque` or `pyvmaf`, and `number of processes` is the number of parallel processes to use to speed up feature extraction. The default `model` is `funque` and the default number of processes is `1`.

2. To train FUNQUE or PyVMAF on your own database, and test on a set of databases, change the training database in `train_models.sh`, modify the list of test databases in `resource/dataset/multi_test_datasets.py`, and run
```
./train_test_models.sh <model> <number of processes>
```

3. `run_sectionwise_selection_and_multitesting.py` shows the constrained feature selection method used to select FUNQUE's features. `run_training.py` trains models, while `run_testing.py` and `run_multi_testing.py` test models on a single, and multiple, test databases respectively.
Finally, `run_funque.py` runs a trained model on a given pair of videos. Use `--help` for more information about their command-line arguments.

4. Database files for all public databases used to evaluate FUNQUE have been provided under `resources/dataset`, and the feature files defining FUNQUE and PyVMAF are available under `resources/feature_param`.

5. Code implementing the unified quality evaluators is available under `funque/third_party/funque_atoms/`, and the Python implementation of VMAF features is available under `funque/third_party/vmaf_atoms`. Both feature extractors are available in `funque/core/custom_feature_extractors.py`.

# References
[1] [https://www.github.com/Netflix/vmaf](https://www.github.com/Netflix/vmaf).

[2] K. Gu, G. Zhai, X. Yang and W. Zhang, "Self-adaptive scale transform for IQA metric," 2013 IEEE International Symposium on Circuits and Systems (ISCAS), Beijing, 2013, pp. 2365-2368, doi: 10.1109/ISCAS.2013.6572353.

[3] A. K. Venkataramanan, C. Wu, A. C. Bovik, I. Katsavounidis and Z. Shahid, "A Hitchhiker’s Guide to Structural Similarity," in IEEE Access, vol. 9, pp. 28872-28896, 2021, doi: 10.1109/ACCESS.2021.3056504.

[4] [https://www.github.com/utlive/enhanced_ssim](https://www.github.com/utlive/enhanced_ssim).

[5] F. Zhang, A. Katsenou, C. Bampis, L. Krasula, Z. Li and D. Bull, "Enhancing VMAF through New Feature Integration and Model Combination," 2021 Picture Coding Symposium (PCS), 2021, pp. 1-5, doi: 10.1109/PCS50896.2021.9477458.
