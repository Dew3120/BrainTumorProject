# Artifact, governance, and experiment appendix

This appendix is generated from committed manifests and registries by [`docs/build_stage9_appendix.py`](build_stage9_appendix.py). It indexes evidence; it does not define new results.

## Frozen stack

- Package status: `FROZEN`
- Manifest: [`runs/final_stack/manifest.json`](../runs/final_stack/manifest.json)
- Manifest SHA256: `7e67de9f16b183165cf0c065e98111cff318ef616630adf52108ce6a57cacf8a`
- Freeze commit: `c2566214bb921c7185fc8949b3e0376492e7113e`
- Frozen Git tree: `439b4fca4b47f12b173c24eaa039b1258f9ddf03`
- Manifest source commit: `7185bf1ea4096881f5846a62f918618fd5c6961c`
- Stage 9 rule: the frozen tree is read-only and unchanged.

### Frozen selections

| Role | Selection |
|---|---|
| `classifier` | v1 raw probabilities |
| `segmenter` | E08 arithmetic mean: E05 + E06 + classifier-routed E07/v1 |
| `calibrator` | C03 final-all-validation isotonic mapping |
| `reliability` | R02 10-feature L2 logistic regression |
| `ood` | O05 secondary logistic ensemble |
| `policy` | O06 ACCEPT/REVIEW; REFUSE disabled |

### Frozen thresholds

| Threshold | Exact value |
|---|---:|
| `segmentation` | `0.50` |
| `reliability` | `0.9490250173235835` |
| `ood_review` | `0.36854275476100085` |
| `ood_refuse` | disabled / null |

### Frozen artifact inventory

| Relative path | Role | Bytes | SHA256 |
|---|---|---:|---|
| `calibration/c03_isotonic_mappings.json` | Selected C03 operational all-validation isotonic mapping | 160,380 | `fdc1a735d78af8e8eb89392e696dc97fe6632729b44de6cf9e9e2d888ea760fa` |
| `configs/c03_result.json` | Configuration, runtime, or policy provenance | 48,020 | `3af002eacbc05e14134abba67135515a58e060b0c4f255bbccd8f10b79c49cb9` |
| `configs/classifier_v1_config.json` | Configuration, runtime, or policy provenance | 1,637 | `940456936632e45ff477c3379cb3a6394728f674e4c88122800cb2707d0245ce` |
| `configs/e08_selection.json` | Configuration, runtime, or policy provenance | 9,341 | `dc2bb7f305d97830b499dc2599050e52c7119c4a3b1ae4997c240cc73edb11b9` |
| `configs/o05_result.json` | Configuration, runtime, or policy provenance | 11,656 | `434d389673aad234ae7e29cb0a96f5f305945e98558c6af3ca41a8b5bb60e7f6` |
| `configs/segmenter_e05_config.json` | Configuration, runtime, or policy provenance | 2,370 | `feb5ad3b4b3fff89c5a87c89a67e4c095a53e01f812dd31dc02e1a7260f7c29c` |
| `configs/segmenter_e06_config.json` | Configuration, runtime, or policy provenance | 2,439 | `0ab097e214b9ca99f5432f636e3089ef5f443b5dad71f2ac732ad260e0e87908` |
| `configs/segmenter_e07_config.json` | Configuration, runtime, or policy provenance | 2,365 | `204f3b5e966822bc22c2a5d3f77ce988d6b9f292985b1510e560ec2f36e7b4c7` |
| `configs/segmenter_v1_config.json` | Configuration, runtime, or policy provenance | 2,134 | `cf2dc82374cae50b3bb16a535615927ac29eac83294bbefeb4d3066165b32b96` |
| `environment/pyproject.toml` | Configuration, runtime, or policy provenance | 746 | `b54b9e20b946e5136a02c335c6120406bdaf1bed6dc2e609b45bd1d2ad787834` |
| `environment/uv.lock` | Configuration, runtime, or policy provenance | 248,136 | `b023b36a7d9cd5c0c691bc8fdf9ed8e2a4d348eb6607ffcc3824bfe26c3effe9` |
| `models/classifier_v1.pt` | Selected v1 four-class classifier; raw softmax probabilities | 44,791,499 | `3bc044027b83b2de225ebde3fb817be8827d2b427c599824697b13ff7a727535` |
| `models/segmenter_e05_weighted.pt` | E08 ensemble component; not selected standalone | 1,624,529 | `e17e7f5afec1c56e9faf12e12db263eff4a3ebc184cfcd7bdfcfff0c1a977c6a` |
| `models/segmenter_e06_boundary.pt` | E08 ensemble component; not selected standalone | 1,624,657 | `b98afabc8118471e01f560ff6a727ce25c08001574bf12005d77d1be773f43ab` |
| `models/segmenter_e07_glioma_specialist.pt` | E08 glioma route component; not selected standalone | 1,624,261 | `bb68ce4f0a9203d74a5ec9dea2e83468d3d833ac285a13ebbdb1d793a0d3bbe0` |
| `models/segmenter_v1_fallback.pt` | E08 non-glioma route and O03 bottleneck source | 1,624,337 | `16dc8745993eef959d0f2924c9147a45320673f6fc0b465ab08f8ad7b2f723de` |
| `ood/message_catalogue.json` | Configuration, runtime, or policy provenance | 1,639 | `bb20ec0abc700c4d6d5c71433d7f8103c2a5b94e513c13c9592e819173ac2a4e` |
| `ood/o02_state.json` | Generated final-stack verification artifact | 1,397 | `c3b9cde754b2f080cf6cad974f72233c050b81d90ebb8697d9a93741f1137159` |
| `ood/o03_mahalanobis_state.npz` | Generated final-stack verification artifact | 221,652 | `38d79dd606e5e800c09ad5a8ce8a11e975e606c289a3daa26dca0e0ca2f7791c` |
| `ood/o03_recovery.json` | Generated final-stack verification artifact | 429 | `9b451cdea9b593ac25bc30ee245e78cf028a2d6ed5c68889e2dd8fb946291134` |
| `ood/o05_normalization_reference.npz` | Generated final-stack verification artifact | 13,397 | `de1a43c6a536db221ddf85ebff78dd347848c6193d5e3ec54b9b4fe6692b0354` |
| `ood/o05_state.json` | Generated final-stack verification artifact | 682 | `58ad42aa7a6f1aef1baea6a260a50eebfff385dbabe176b1d9f6e4231d125b02` |
| `ood/o06_policy.json` | Configuration, runtime, or policy provenance | 3,576 | `44f71a31feb09d412b4e25279bc8a98f108baa4cef9049fcdba9ea9cee65b8be` |
| `reliability/r02_reliability_logistic.pkl` | Selected R02 deployment logistic model | 736 | `9f8ea8530c1c2ee6a0773c6ec7c0b92e6da7923ddf3c50d3d32176379339f044` |
| `reliability/r02_reliability_manifest.json` | R02 feature order, scaler state, coefficients, and provenance | 4,901 | `0d56aab5d562a34325003cee116c4f9d5f8fbe3fcc5278717338706d58e5303f` |
| `reliability/r02_reliability_scaler.pkl` | Selected R02 deployment scaler | 648 | `c99a67f8157515d815a9b7a05555b45e26b5a2da0e458b359cc83d63fe30dce8` |
| `runtime/final_stack_runtime.py` | Configuration, runtime, or policy provenance | 17,885 | `0f563801329454a35fe46bfb40c855501c76f253d56d898ddbf7160dd54b4309` |
| `verification/expected_outputs.json` | Generated final-stack verification artifact | 5,128 | `7b94b8a84adcd6da8d097491f2b21f5905c68072dbe5f99783f644c18e51b17c` |
| `verification/fixtures/elliptic.png` | Generated final-stack verification artifact | 848 | `fbf404c68f5dc373656ea55b1724926b3192407cf8068ac820c952027dfc92f7` |
| `verification/fixtures/gradient.png` | Generated final-stack verification artifact | 735 | `b26228576c7fe2ba78089c8ba81dd6a36b1de2e664e06578dd09cb013ab4c606` |
| `verification/fixtures/noise.png` | Generated final-stack verification artifact | 33,353 | `c784d36a24c892c1653f7e06bbe8248b19a75ed063771e3715b750597198f6ce` |

## Amendments

The amendment sequence is preserved in numerical order. Each file was committed before the authorized downstream action continued.

| Amendment | Title | SHA256 |
|---|---|---|
| [preregistration_amendment_01](../audit/preregistration_amendment_01.md) | Pre-registration amendment 01 — glioma improvement and reliability program | `62abb1d13d3ffd16d3fdf41b461ad48aafc746caf4639f18e4545467ebf8c0fb` |
| [preregistration_amendment_02](../audit/preregistration_amendment_02.md) | Pre-registration amendment 02 — conditional E08 final-model pathway | `0153a8709f16ee38e4e2e2cea6044c840bb8b775971e2d090a02537a473e6a43` |
| [preregistration_amendment_03](../audit/preregistration_amendment_03.md) | Pre-registration amendment 03 — calibrated E08 reliability program | `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40` |
| [preregistration_amendment_04](../audit/preregistration_amendment_04.md) | Pre-registration amendment 04 — R02 reduced reliability model | `1603be8a4d6b4415ebac4162260968def6b9c539ec03bb67c18d7bb4016f9629` |
| [preregistration_amendment_05](../audit/preregistration_amendment_05.md) | Pre-registration amendment 05 — R03 extended reliability model | `a5c8ff14dea6349c7b3be334e27edf6323f1e0e23a772f530223e89f6bfc03a3` |
| [preregistration_amendment_06](../audit/preregistration_amendment_06.md) | Pre-registration amendment 06 — exact R03 image-transform dtype | `47ee34350db79c44d0a5d74bf9cb809029b8781f930959c030705e03e4e4b394` |
| [preregistration_amendment_07](../audit/preregistration_amendment_07.md) | Pre-registration amendment 07 — R04 per-category reliability models | `cd960ba0167f3982722667651a2dd0da12e2c2437160df7b6c0cee1f65be2a40` |
| [preregistration_amendment_08](../audit/preregistration_amendment_08.md) | Pre-registration amendment 08 — R05 category-aware reliability model | `3864f8dab879d8bf7821b91551127380fd636d1b68a170193721ccba860223c0` |
| [preregistration_amendment_09](../audit/preregistration_amendment_09.md) | Pre-registration amendment 09 — R06 nonlinear reliability comparison | `84305f4e04404d238407cc8650317d111aa871c72b67d62844cdd7ece921cdb7` |
| [preregistration_amendment_10](../audit/preregistration_amendment_10.md) | Pre-registration amendment 10 — R07 coefficient bootstrap | `2f96675d8616605cb9794928e4b7dfa1fd9a76361f09d9ddcd802fa7f86ed4cc` |
| [preregistration_amendment_11](../audit/preregistration_amendment_11.md) | Pre-registration amendment 11 — R08 prospective fold evaluation | `79e3e5d49230b4ec0409f98842b2e40d836981aa82e34e6269a7589f58d9ad7a` |
| [preregistration_amendment_12](../audit/preregistration_amendment_12.md) | Pre-registration amendment 12 — R09 fold sensitivity | `5902598fae407dcd56bb0084c01098974e0cc78e20fb406d8d733dbf7015ec7a` |
| [preregistration_amendment_13](../audit/preregistration_amendment_13.md) | Pre-registration amendment 13 — external cross-dataset transport | `728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5` |
| [preregistration_amendment_14](../audit/preregistration_amendment_14.md) | Pre-registration amendment 14 — deployable R02 reliability estimator | `8abe7d0fb56c116f2c3f3f667e0263564dcb89b5021b8a3dc61959487849b7d5` |
| [preregistration_amendment_15](../audit/preregistration_amendment_15.md) | Pre-registration amendment 15 — X02 target representability and restart | `4fea798e9a50ef12d5ea2d9cc627cfd8594207860cba10f3bb82e2b7ec782d9e` |
| [preregistration_amendment_16](../audit/preregistration_amendment_16.md) | Pre-registration amendment 16 — UPENN-GBM curated derivative identity and label encoding | `802ade9daecfdbc11fcd9344c7aee28e9199562f46fc1990624c940ad8b6bf34` |
| [preregistration_amendment_17](../audit/preregistration_amendment_17.md) | Pre-registration amendment 17 — X03 target representability and one restart | `3c7941e388f8c095884d7bd62bc1886fb3aa8a1b5003520f76cdcbffd6ee12b7` |
| [preregistration_amendment_18](../audit/preregistration_amendment_18.md) | Pre-registration amendment 18 — preserve native NIfTI datatype and authorize X03 attempt three | `cce55327a687c6ef4adec4ba85284e887438bfc6352abeb615f0ab3f361d8c63` |
| [preregistration_amendment_19](../audit/preregistration_amendment_19.md) | Pre-registration amendment 19 — X03 domain-bootstrap membership correction and attempt four | `ab9ba9bcab3896a359e94bf6c72887c8bb69b785cfd62aff0b17ab58b8ec0f66` |
| [preregistration_amendment_20](../audit/preregistration_amendment_20.md) | Pre-registration amendment 20 — out-of-distribution detection | `52b7d3e99a7354b2dafc42de0fac8d97155b61ddf5fa938a91189f91a92c863b` |

## Experiment index

The registry can retain umbrella or future experiments as `REGISTERED_NOT_STARTED`; that status is not represented as completed work.

| ID | Status | Hypotheses | Package |
|---|---|---|---|
| `glioma_segmentation_improvement` | `REGISTERED_NOT_STARTED` | H1 | [`experiments/glioma_segmentation_improvement`](../experiments/glioma_segmentation_improvement/) |
| `e01_lcc_filter` | `COMPLETE` | H1 | [`experiments/e01_lcc_filter`](../experiments/e01_lcc_filter/) |
| `e02_morphology` | `COMPLETE` | H1 | [`experiments/e02_morphology`](../experiments/e02_morphology/) |
| `e03_tta` | `COMPLETE` | H1 | [`experiments/e03_tta`](../experiments/e03_tta/) |
| `e04_multiscale` | `COMPLETE` | H1 | [`experiments/e04_multiscale`](../experiments/e04_multiscale/) |
| `e05_glioma_weighted` | `COMPLETE` | H1 | [`experiments/e05_glioma_weighted`](../experiments/e05_glioma_weighted/) |
| `e06_boundary_loss` | `COMPLETE` | H1 | [`experiments/e06_boundary_loss`](../experiments/e06_boundary_loss/) |
| `e07_glioma_specialized` | `COMPLETE` | H1 | [`experiments/e07_glioma_specialized`](../experiments/e07_glioma_specialized/) |
| `e08_ensemble` | `COMPLETE` | H1 | [`experiments/e08_ensemble`](../experiments/e08_ensemble/) |
| `c01_vector_scaling` | `COMPLETE` | H3 | [`experiments/c01_vector_scaling`](../experiments/c01_vector_scaling/) |
| `c02_dirichlet` | `COMPLETE` | H3 | [`experiments/c02_dirichlet`](../experiments/c02_dirichlet/) |
| `c03_isotonic` | `COMPLETE` | H3, H7 | [`experiments/c03_isotonic`](../experiments/c03_isotonic/) |
| `c04_split_temperature` | `COMPLETE` | H3 | [`experiments/c04_split_temperature`](../experiments/c04_split_temperature/) |
| `c05_reliability_weighted` | `COMPLETE` | H3, H7 | [`experiments/c05_reliability_weighted`](../experiments/c05_reliability_weighted/) |
| `c06_per_category` | `COMPLETE` | H3, H7 | [`experiments/c06_per_category`](../experiments/c06_per_category/) |
| `c07_hierarchical` | `COMPLETE` | H3, H7 | [`experiments/c07_hierarchical`](../experiments/c07_hierarchical/) |
| `c08_unified_ece` | `COMPLETE` | H3 | [`experiments/c08_unified_ece`](../experiments/c08_unified_ece/) |
| `c09_category_diagrams` | `COMPLETE` | H3 | [`experiments/c09_category_diagrams`](../experiments/c09_category_diagrams/) |
| `r01_ablation` | `COMPLETE` | H4, H7 | [`experiments/r01_ablation`](../experiments/r01_ablation/) |
| `r02_reduced_features` | `COMPLETE` | H4, H7 | [`experiments/r02_reduced_features`](../experiments/r02_reduced_features/) |
| `r03_extended_features` | `COMPLETE` | H4, H7 | [`experiments/r03_extended_features`](../experiments/r03_extended_features/) |
| `r04_per_category` | `COMPLETE` | H4, H7 | [`experiments/r04_per_category`](../experiments/r04_per_category/) |
| `r05_category_aware` | `COMPLETE` | H4, H7 | [`experiments/r05_category_aware`](../experiments/r05_category_aware/) |
| `r06_nonlinear_comparison` | `COMPLETE` | H4, H7 | [`experiments/r06_nonlinear_comparison`](../experiments/r06_nonlinear_comparison/) |
| `r07_coefficient_bootstrap` | `COMPLETE` | H4, H7 | [`experiments/r07_coefficient_bootstrap`](../experiments/r07_coefficient_bootstrap/) |
| `r08_prospective_fold_evaluation` | `COMPLETE` | H4, H7 | [`experiments/r08_prospective_fold_evaluation`](../experiments/r08_prospective_fold_evaluation/) |
| `r09_fold_sensitivity` | `COMPLETE` | H4, H7 | [`experiments/r09_fold_sensitivity`](../experiments/r09_fold_sensitivity/) |
| `r10_seed_stability` | `COMPLETE` | H4, H7 | [`experiments/r10_seed_stability`](../experiments/r10_seed_stability/) |
| `e08_final_selection` | `COMPLETE` | H7 | [`experiments/e08_final_selection`](../experiments/e08_final_selection/) |
| `p01_cost_model` | `COMPLETE` | — | [`experiments/p01_cost_model`](../experiments/p01_cost_model/) |
| `p02_threshold_selection` | `COMPLETE` | — | [`experiments/p02_threshold_selection`](../experiments/p02_threshold_selection/) |
| `p03_decision_rule` | `COMPLETE` | — | [`experiments/p03_decision_rule`](../experiments/p03_decision_rule/) |
| `p04_warning_copy` | `COMPLETE` | — | [`experiments/p04_warning_copy`](../experiments/p04_warning_copy/) |
| `p05_threshold_sensitivity` | `COMPLETE` | — | [`experiments/p05_threshold_sensitivity`](../experiments/p05_threshold_sensitivity/) |
| `x00_r02_deployable_model` | `COMPLETE` | H8 | [`experiments/x00_r02_deployable_model`](../experiments/x00_r02_deployable_model/) |
| `x01_cheng_overlap_control` | `COMPLETE` | H8 | [`experiments/x01_cheng_overlap_control`](../experiments/x01_cheng_overlap_control/) |
| `x02_openneuro_pituitary` | `COMPLETE` | H8 | [`experiments/x02_openneuro_pituitary`](../experiments/x02_openneuro_pituitary/) |
| `x03_upenn_gbm` | `COMPLETE` | H8 | [`experiments/x03_upenn_gbm`](../experiments/x03_upenn_gbm/) |
| `x04_meningioma_seg_class` | `ACCESS_BLOCKED` | H8 | [`experiments/x04_meningioma_seg_class`](../experiments/x04_meningioma_seg_class/) |
| `x05_cross_dataset_synthesis` | `COMPLETE` | H8 | [`experiments/x05_cross_dataset_synthesis`](../experiments/x05_cross_dataset_synthesis/) |
| `sealed_final_evaluation` | `REGISTERED_NOT_STARTED` | H2, H3, H4 | [`experiments/sealed_final_evaluation`](../experiments/sealed_final_evaluation/) |
| `application_integrity_validation` | `REGISTERED_NOT_STARTED` | H5 | [`experiments/application_integrity_validation`](../experiments/application_integrity_validation/) |
| `measurement_engine_validation` | `REGISTERED_NOT_STARTED` | H6 | [`experiments/measurement_engine_validation`](../experiments/measurement_engine_validation/) |
| `o01_ood_benchmark` | `COMPLETE` | H9 | [`experiments/o01_ood_benchmark`](../experiments/o01_ood_benchmark/) |
| `o02_feature_ood` | `COMPLETE` | H9 | [`experiments/o02_feature_ood`](../experiments/o02_feature_ood/) |
| `o03_encoder_mahalanobis` | `COMPLETE` | H9 | [`experiments/o03_encoder_mahalanobis`](../experiments/o03_encoder_mahalanobis/) |
| `o04_model_confidence_ood` | `COMPLETE` | H9 | [`experiments/o04_model_confidence_ood`](../experiments/o04_model_confidence_ood/) |
| `o05_ensemble_ood` | `COMPLETE` | H9 | [`experiments/o05_ensemble_ood`](../experiments/o05_ensemble_ood/) |
| `o06_ood_policy` | `COMPLETE` | H9 | [`experiments/o06_ood_policy`](../experiments/o06_ood_policy/) |

## Principal audit reports

- [Pre-registration](../audit/preregistration.md)
- [Review log](../audit/review_log.md)
- [Glioma failure findings](../audit/glioma_failure_findings.md)
- [Calibration findings](../audit/confidence_calibration_findings.json)
- [Stage 3 completion](../audit/stage3_completion_report.md)
- [Stage 4 completion](../audit/stage4_completion_report.md)
- [Stage 5 completion](../audit/stage5_completion_report.md)
- [Stage 6 cross-dataset synthesis](../audit/stage6_cross_dataset.md)
- [Stage 7 OOD completion](../audit/stage7_completion_report.md)
- [Stage 8 final freezing](../audit/stage8_final_freezing.md)
- [Stage 9 metric source ledger](stage9_source_ledger.json)

## Figure index

Every figure is generated or wrapped at 300 DPI by [`docs/build_stage9_figures.py`](build_stage9_figures.py). Source hashes are recorded in [`docs/figures/figure_manifest.json`](figures/figure_manifest.json).

| Figure | Description | SHA256 |
|---|---|---|
| [`01_pipeline_architecture.png`](figures/01_pipeline_architecture.png) | Frozen inference architecture | `683035a5c53090e85cd06fd9111dab87cd43be33b341d0e417efe97d98faa5b2` |
| [`02_classifier_confusion_matrix.png`](figures/02_classifier_confusion_matrix.png) | Classifier internal-validation confusion matrix | `f858df13b781b3e66a0dbf76c17678fa2ab21bbc5929e89440ca4463ad9778fd` |
| [`03_segmentation_examples.png`](figures/03_segmentation_examples.png) | E08 success and failure examples | `0f234d63dcda3b09c7486628ea56724220b55673cb1331eed5c2eb54e066fb80` |
| [`04_c03_reliability.png`](figures/04_c03_reliability.png) | Raw and calibrated pixel reliability | `c9c9c9951752f1e1a9081ac1bb8c9d55b2e3f07805ff3ba5e967ff25b6ae35c3` |
| [`05_reliability_roc.png`](figures/05_reliability_roc.png) | Reliability ROC | `ef723e6ea71feef4ae6844817ca6e4bf0848d58ac58c125017d6e93737dbc51d` |
| [`06_ood_roc.png`](figures/06_ood_roc.png) | OOD ROC | `107381fdd2d735a8fe5217b3f8d9f4ba62eb05f16271dd4a2dffdad1c9ed1fa7` |
| [`07_cross_dataset_transport.png`](figures/07_cross_dataset_transport.png) | External transport summary | `beac28775e0459be68a4ba486eb7c68566d84fed1e4ea8975fc5e9b8290edf29` |
| [`08_failure_mode_breakdown.png`](figures/08_failure_mode_breakdown.png) | Failure-mode proportions | `0e13bce6003af69466d564d822a3b3cef12469f2aea4885865dc94dddb4dfaf2` |
| [`09_calibration_comparison.png`](figures/09_calibration_comparison.png) | C01-C07 calibration comparison | `38aeabf6bd6cbb061ae2a9681b3d3823b6aef97383bad209c8cdfb42fc6b7734` |
| [`10_feature_importance.png`](figures/10_feature_importance.png) | R01 feature importance | `593a0a29f1c64327a4b64bf68b9c3506a9e7c0c5f99fbb9c235cf491698ffc7d` |
| [`11_threshold_sensitivity.png`](figures/11_threshold_sensitivity.png) | Threshold sensitivity | `7546c8b7bc1ba2ac449a540c52dd15314c9b1df42549677e014692e62b13ff7c` |
| [`12_stage_progression_timeline.png`](figures/12_stage_progression_timeline.png) | Project-stage timeline | `882882310050458741793c6d7c6fdc348acdb41dfdde574c9b7b3e23f30e5de3` |
| [`13_policy_cost_curve.png`](figures/13_policy_cost_curve.png) | Expected-cost curve | `9885d4d88489d44139c7a81d349754275a1d5d0aa43c502378033a5198e894f1` |
| [`14_reliability_transfer.png`](figures/14_reliability_transfer.png) | Reliability transfer AUROC | `d04005f5a79745aab5c5c148227aef88ee229a7bcb3f580c03e0c6fe186519f8` |

## Reproduction commands

```powershell
uv run python docs/build_stage9_figures.py
uv run python docs/build_stage9_appendix.py
uv run python docs/verify_stage9_docs.py
uv run python scripts/verify_final_stack.py
uv run ruff check .
$env:PYTHONPATH=((Resolve-Path '.').Path + ';' + (Resolve-Path 'scripts').Path)
uv run pytest -q
```

Regenerating documentation must not write to `runs/final_stack/`. The final verification checks its Git tree and manifest hash.

## Claim scope

All metrics are retrospective research results. BRISC values are internal development-validation results with patient independence unverified. H8 was rejected on the evaluated independent cohorts. The OOD benchmark is registered but source-specific. Nothing in this appendix establishes clinical validity.
