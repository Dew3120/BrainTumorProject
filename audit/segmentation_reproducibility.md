# Segmentation reproducibility audit

## Determinism conclusion

**Controlled, but not fully guaranteed.** The original training run seeded Python, NumPy, Torch CPU/CUDA, the shuffle generator, and workers. It disabled cuDNN benchmarking, enabled cuDNN deterministic mode, used no augmentation, and used zero dropout. Data order and preprocessing are therefore controlled on the recorded stack.

The run did not call `torch.use_deterministic_algorithms(True)`, did not record `CUBLAS_WORKSPACE_CONFIG`, and did not record a TF32 policy. A complete 68-epoch replay was intentionally not performed because reproducibility audit prohibits retraining. The current CUDA stack passed two bitwise-identical synthetic forward/backward/AdamW/AMP steps with PyTorch's strict deterministic guard enabled; this is an operation-path check, not proof that the historical full run is bitwise reproducible everywhere.

| Control | Recorded |
|---|---:|
| Python/NumPy/Torch CPU/Torch CUDA seed 42 | True |
| cuDNN benchmark disabled | True |
| cuDNN deterministic mode enabled | True |
| Shuffle generator and workers seeded | True |
| Augmentation disabled and dropout zero | True |
| Global deterministic-algorithm guard used in original run | False |
| CUBLAS workspace policy recorded | False |
| TF32 policy recorded | False |

Specific remaining sources of uncertainty:

- **CUDA Conv2d and ConvTranspose2d forward/backward through cuDNN:** cuDNN deterministic mode was enabled, but the original run did not enable PyTorch's global deterministic-algorithm guard. An unsupported nondeterministic path would not have failed closed.
- **InstanceNorm2d, Dice/BCE losses, and CUDA gradient reductions:** Parallel floating-point reduction order and kernel implementation can change across CUDA, cuDNN, PyTorch, driver, or GPU versions. The current runtime probe passes, but cross-runtime bitwise identity is not guaranteed.
- **CUDA AdamW update with AMP FP16 forward and GradScaler:** FP16 rounding and overflow decisions can amplify small runtime-level differences. Early stopping and top-3 selection can then follow a different path even with the same seed.
- **CUDA library determinism enforcement:** CUBLAS_WORKSPACE_CONFIG and TF32 policy were not explicitly recorded by the training run. The model path is convolution-heavy, but the environment contract is incomplete.

## Checkpoint reload

- Accepted checkpoint: `E:\BrainTumorProject\runs\segmentation\20260923-072439-397263-full\top3_checkpoints\epoch_056_macro_dice_0.762662.pt`
- Checkpoint SHA256: `16DC8745993EEF959D0F2924C9147A45320673F6FC0B465AB08F8AD7B2F723DE`
- Validation images recomputed: 680
- Maximum aggregate metric difference: 0.0
- Maximum per-image Dice/IoU difference: 0.0
- Test images accessed: 0

This restates and independently repeats the successful checkpoint reload: epoch 56 reproduces every stored validation metric exactly on the current CUDA runtime.

## Run manifest

- Manifest: `E:\BrainTumorProject\runs\segmentation\20260923-072439-397263-full\segmentation_run_manifest.json`
- Manifest SHA256: `b837985f4df75ca3d2b43e1ae318254d5227037ac1d856a4ede2020b68c21a7a`
- Hashed run files: 25
- Source-equivalent training commit: `459c67555851edb950c6304492beca926b995eb1`
- Manifest audit base commit: `56e96d9b3d67cebee1e938fff8e769f704d9c233`

The exact training-time Git HEAD was not recorded by the historical run. Commit `459c675` is the first repository commit containing normalized-text-identical copies of all three saved training sources and the accepted segmentation findings. The manifest excludes itself because a file cannot contain its own final SHA256; its external SHA256 is recorded above and in MLflow.

## Positional-bias follow-up

A consistent skull-base positional bias was not confirmed across all 17 zero-Dice cases. Errors were spatially heterogeneous, with a modest lower-half shift: 11/17 centroids were below the image midline and median normalized y was 0.534 versus 0.463 among nonzero-Dice cases; only 1/17 centroids fell below y=0.60 and the median lower-third pixel fraction was 0.008.

The consolidated findings must also retain the tumor-type confound (12/17 zero-Dice cases are glioma while 15/20 highest-Dice cases are meningioma) and quote the uncertainty-analysis Spearman correlations directly: classifier confidence versus segmenter Dice `0.3252`, and classifier confidence versus segmenter inside-reference probability `0.1660`.

## Scope

No model was retrained, no threshold changed, no test image was accessed, and calibration and uncertainty work was not started.
