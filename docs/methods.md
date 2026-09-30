# Methodology and metric derivations

This document defines the methods used by the frozen brain-tumor MRI research pipeline. It is a technical companion to [the narrative report](report.md). Exact operational state is defined by [the frozen manifest](../runs/final_stack/manifest.json); this document does not override that manifest.

## 1. Notation

For a binary reference mask, let \(y_i\in\{0,1\}\) denote pixel \(i\), \(p_i\in[0,1]\) the predicted foreground probability, and \(\hat y_i=\mathbb{1}[p_i\ge 0.50]\) the hard prediction. There are \(N\) pixels in one image. For case-level reliability, let \(z_j=1\) if case \(j\) reaches Dice at least 0.50 and \(z_j=0\) otherwise. Let \(r_j\) be R02’s estimated probability of success.

For multiclass classification, \(K=4\), \(y_{jk}\) is a one-hot class indicator, and \(q_{jk}\) is the predicted probability for class \(k\). Class order is fixed as glioma, meningioma, pituitary, and no tumor.

## 2. Data partition and grouping

The project-created BRISC development split uses training, validation, sealed test, and review-excluded assignments. Classification training uses 3,570 images and internal validation uses 893. Segmentation uses 2,719 tumor-positive training pairs and 680 internal validation pairs. The 1,000 supplied classification test images and 860 tumor-positive test pairs remain sealed.

Known or suspected similar images are assigned to common groups. Cross-fitting uses `StratifiedGroupKFold` so a registered group does not cross fit/evaluation folds. Stratification preserves failure prevalence as far as possible while respecting groups. This is similarity-aware, not patient-verified: BRISC does not provide complete identifiers that establish patient independence.

External confidence intervals resample subjects when stable subject identifiers exist. Internal intervals resample similarity groups. A bootstrap draw includes all rows belonging to each sampled group, including repeated draws of a group.

## 3. Image preprocessing

### 3.1 Classifier

1. Decode the image and convert it to grayscale.
2. Resize while preserving aspect ratio so the image fits inside 224×224.
3. Center the resized image on a black 224×224 canvas.
4. Repeat the grayscale channel to RGB.
5. Divide by 255 when required by the decoded dtype path.
6. Normalize by ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

Training adds brightness and contrast jitter of 0.1. Validation and inference do not. No geometric classifier augmentation is used.

### 3.2 Segmenter

1. Decode as single-channel grayscale while preserving the registered native-dtype path.
2. Resize while preserving aspect ratio so the image fits inside 128×128.
3. Center on a black 128×128 canvas.
4. Scale intensity according to the frozen adapter contract.
5. Convert to a one-channel tensor.

Masks follow the same geometry but use nearest-neighbor interpolation and binary values. The hard mask is defined by probability at least 0.50. The adapter is fixed; no mask dilation or post-hoc resize correction is applied.

### 3.3 Source-coordinate restoration

For external analysis, a model-space hard mask can be restored into the unpadded resized region and then resized by nearest neighbor to the native image dimensions. Source-coordinate metrics measure that restored prediction against the native reference. These metrics do not alter the primary strict model-space analysis.

## 4. Classifier

The classifier is a ResNet-18 initialized with ImageNet-1K v1 weights. Its final fully connected layer outputs four logits \(a_{jk}\). Raw softmax probabilities are

\[
q_{jk}=\frac{\exp(a_{jk})}{\sum_{l=1}^{K}\exp(a_{jl})}.
\]

The predicted class is \(\arg\max_k q_{jk}\). Raw probabilities are operational because registered temperature-scaling studies did not establish a stable improvement.

### 4.1 Precision, recall, and F1

For class \(k\), with one-versus-rest counts \(TP_k,FP_k,FN_k\),

\[
\operatorname{precision}_k=\frac{TP_k}{TP_k+FP_k},\qquad
\operatorname{recall}_k=\frac{TP_k}{TP_k+FN_k},
\]

and

\[
F1_k=\frac{2\operatorname{precision}_k\operatorname{recall}_k}
{\operatorname{precision}_k+\operatorname{recall}_k}.
\]

Macro F1 is \(K^{-1}\sum_k F1_k\). It weights each category equally regardless of category count.

## 5. Segmentation models

### 5.1 v1 U-Net

The v1 segmenter is a compact 2D MONAI U-Net with channels `(16, 32, 64, 128)`, two residual units, one input channel, one output logit channel, and 401,277 parameters. The output head is linear; sigmoid is applied only to obtain probabilities.

### 5.2 Balanced binary cross-entropy

For each training image, the foreground weight is derived from foreground/background counts. With weight \(w_+\), weighted BCE with logits is

\[
L_{BCE}=-\frac{1}{N}\sum_i\left[w_+y_i\log \sigma(s_i)
+(1-y_i)\log(1-\sigma(s_i))\right],
\]

where \(s_i\) is the raw logit and \(\sigma\) is sigmoid. The final implementation uses the registered per-image weighting behavior from the frozen training script.

### 5.3 Soft Dice loss

With smoothing constant \(\epsilon\), soft Dice is

\[
D_{soft}=\frac{2\sum_i p_i y_i+\epsilon}
{\sum_i p_i+\sum_i y_i+\epsilon},
\]

and Dice loss is \(1-D_{soft}\). The combined baseline objective is the registered weighted BCE plus foreground Dice term.

### 5.4 E08 ensemble

Let \(p^{05}\), \(p^{06}\), \(p^{07}\), and \(p^{v1}\) be component probability maps. The routed third map is

\[
p^{route}=\begin{cases}
p^{07},&\text{classifier predicts glioma},\\
p^{v1},&\text{otherwise}.
\end{cases}
\]

E08 is

\[
p^{E08}=\frac{p^{05}+p^{06}+p^{route}}{3}.
\]

The hard mask is \(\mathbb{1}[p^{E08}\ge0.50]\). No component receives a learned ensemble weight.

## 6. Spatial metrics

Let \(A=\{i:y_i=1\}\) and \(B=\{i:\hat y_i=1\}\).

### 6.1 Dice

\[
Dice(A,B)=\frac{2|A\cap B|}{|A|+|B|}.
\]

When the reference is non-empty and the prediction is empty, Dice is zero. The external strict analyses also assign zero to a source-positive target erased by the frozen 128×128 mask adapter.

### 6.2 Intersection over union

\[
IoU(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

Dice and IoU are monotonic transforms for one case when both masks are fixed, but their averages differ.

### 6.3 Boundary F1

Reference and predicted boundaries are extracted from binary masks. A predicted boundary pixel is matched if a reference boundary pixel lies within the registered pixel tolerance; reference recall is defined symmetrically. Boundary precision and recall are combined as F1. This metric is sensitive to contour alignment and complements area overlap.

### 6.4 Macro aggregation

Image-mean Dice averages over images. Category Dice averages within each true BRISC category. Macro category Dice averages the three category means equally. External subject-macro Dice first averages slices within subject and then averages subjects, preventing subjects with more slices from dominating.

## 7. Pixel calibration

### 7.1 Negative log-likelihood

Binary pixel NLL is

\[
NLL=-\frac{1}{N}\sum_i[y_i\log(p_i)+(1-y_i)\log(1-p_i)].
\]

Probabilities are clipped only by the registered numerical guard to avoid logarithm of zero.

### 7.2 Brier score

\[
Brier=\frac{1}{N}\sum_i(p_i-y_i)^2.
\]

It is a proper scoring rule that measures squared probability error.

### 7.3 Expected and maximum calibration error

Partition predictions into \(M\) bins \(B_m\). For each bin, let

\[
conf(B_m)=\frac{1}{|B_m|}\sum_{i\in B_m}p_i,
\quad
acc(B_m)=\frac{1}{|B_m|}\sum_{i\in B_m}y_i.
\]

Then

\[
ECE=\sum_{m=1}^{M}\frac{|B_m|}{N}|acc(B_m)-conf(B_m)|,
\]

and

\[
MCE=\max_m|acc(B_m)-conf(B_m)|.
\]

Stage 3’s unified comparisons use 15 equal-frequency bins. Equal-frequency bins make each bin contribute a similar observation count and avoid many nearly empty bins in the extreme pixel-probability distribution.

### 7.4 Temperature scaling

For scalar temperature \(T>0\), calibrated probability is \(\sigma(s_i/T)\). For multiclass logits, softmax uses \(a_{jk}/T\). Vector scaling allows class-specific temperatures. Temperature is fitted by minimizing the registered NLL on fitting folds.

### 7.5 Isotonic regression

Isotonic regression learns a non-decreasing piecewise-constant mapping \(g:[0,1]\rightarrow[0,1]\) by minimizing squared error subject to monotonicity:

\[
\min_g \sum_i (g(p_i)-y_i)^2\quad\text{subject to }p_i\le p_j\Rightarrow g(p_i)\le g(p_j).
\]

C03 fits one mapping per cross-fitting fold and reports held-out mapped probabilities. The frozen operational mapping is fitted once on all 680 internal validation cases and stored as 501 knots. It is explicitly not the mapping used for reported OOF metrics.

## 8. Case-level reliability

### 8.1 Target

The registered target is

\[
z_j=\mathbb{1}[Dice_j\ge0.50].
\]

R02 models \(P(z_j=1\mid x_j)\), where \(x_j\) is the ten-feature prediction-only vector. During failure-detection AUROC calculation, risk is \(1-r_j\) and failure is \(1-z_j\).

### 8.2 Feature definitions

- **empty_prediction:** one when the hard mask contains zero foreground pixels.
- **predicted_area_fraction:** hard-mask foreground pixels divided by all model-space pixels.
- **whole_probability_mean:** mean calibrated foreground probability over the canvas.
- **whole_probability_sd:** standard deviation of calibrated probabilities.
- **predicted_region_entropy_mean:** mean Bernoulli entropy where the hard mask is foreground; fail-closed default follows the frozen feature contract for an empty mask.
- **boundary_entropy_mean:** mean Bernoulli entropy on the hard-mask boundary.
- **high_confidence_foreground_fraction:** registered fraction of pixels exceeding the high-confidence foreground probability rule.
- **uncertain_pixel_fraction:** registered fraction in the probability uncertainty interval.
- **bounding_box_fill_ratio:** hard-mask area divided by the area of its axis-aligned bounding box.
- **normalized_perimeter:** hard-mask perimeter under the registered normalization.

Bernoulli entropy is

\[
H(p)=-p\log p-(1-p)\log(1-p).
\]

### 8.3 Standardization and logistic model

For feature \(k\),

\[
\tilde x_{jk}=\frac{x_{jk}-\mu_k}{\sigma_k}.
\]

The success probability is

\[
r_j=\sigma\left(b+\sum_{k=1}^{10}\beta_k\tilde x_{jk}\right).
\]

The scaler means, scales, coefficient vector, intercept, and exact feature order are in the frozen manifest. L2 regularization uses `C=1.0` and the `lbfgs` solver. Reported metrics use five-fold OOF scores; the serialized deployment estimator is a deterministic full-population fit.

### 8.4 AUROC

AUROC is the probability that a randomly chosen positive-ranking example receives a greater score than a randomly chosen negative-ranking example, with half credit for ties. For failure AUROC, weak masks are positive and risk \(1-r\) is the score.

### 8.5 Brier skill

For case probabilities,

\[
Brier_{model}=\frac{1}{J}\sum_j(r_j-z_j)^2.
\]

The fold-specific baseline predicts the training-fold prevalence for every held-out case, avoiding held-out prevalence leakage. Brier skill is

\[
BS_{skill}=1-\frac{Brier_{model}}{Brier_{baseline}}.
\]

Positive skill means improvement over the registered prevalence baseline.

## 9. Cost-sensitive reliability policy

Define a weak mask as policy-positive for review. Review of either a weak or acceptable case costs 1; accepting an acceptable case costs 0; accepting a weak case costs 10. For threshold \(\theta\), a case is reviewed when \(r<\theta\). Total cost is

\[
C(\theta)=TP_{review}+FP_{review}+10FN_{failure}.
\]

P02 evaluates every distinct OOF reliability score plus endpoints. The selected threshold minimizes total cost, with registered tie-breaking applied before inspection. The selected value is exactly `0.9490250173235835`.

A three-state grid introduced `REVIEW_REQUIRED`, `WARNING`, and `STANDARD` with registered state costs. No non-empty warning band reduced cost, so the final policy collapses to two states. Invalid or missing scores map to `REVIEW_REQUIRED`.

## 10. Out-of-distribution detection

### 10.1 O02 image-statistics probability

O02 computes registered inference-time properties of the standardized image and its source representation. A scaler and L2 logistic model estimate the probability that the input belongs to the registered OOD side of the fitting benchmark. Because encoding and file statistics are powerful, O02 may capture dataset source.

### 10.2 O03 Mahalanobis distance

Let \(h\) be the frozen v1 segmenter bottleneck embedding, \(\mu\) the BRISC fitting mean, and \(\Lambda\) the Ledoit–Wolf precision matrix. Squared Mahalanobis distance is

\[
d^2(h)=(h-\mu)^T\Lambda(h-\mu).
\]

Ledoit–Wolf shrinkage stabilizes covariance estimation in a high-dimensional embedding space by shrinking the empirical covariance toward a structured target. The Stage 8 reconstruction regenerated \(\mu\) and \(\Lambda\) from the frozen 600 fitting embeddings and reproduced recorded distances within 1e-12.

### 10.3 O04 uncertainty scores

Classifier uncertainty is based on \(1-\max_k q_k\). Segmenter uncertainty is mean Bernoulli entropy across the model-space probability map. Both failed their standalone AUROC gates but contribute to O05.

### 10.4 Empirical-CDF normalization

For component score \(s\), the empirical CDF based on the 600 BRISC fitting scores is

\[
F_{ID}(s)=\frac{1}{600}\sum_{j=1}^{600}\mathbb{1}[s_j\le s].
\]

The four normalized scores are standardized and passed to the O05 secondary logistic regression. The OOD score is the resulting probability-like output. Its calibration is benchmark-specific.

### 10.5 O06 policy

The review threshold is the registered 95th percentile of BRISC fitting scores: `0.36854275476100085`. OOD score at or above the threshold gives `REVIEW`; below gives `ACCEPT`. REFUSE is disabled. Invalid values fail closed to review.

## 11. Bootstrap confidence intervals

Unless an experiment specifies otherwise, the project uses 2,000 bootstrap draws with seed 42. A group-bootstrap draw samples group identifiers with replacement and concatenates all member cases. The metric is recomputed on the sampled rows. The 2.5th and 97.5th percentiles form the percentile 95% CI.

For paired model differences, the same draw is used for both models and the metric difference is computed within the draw. This preserves case pairing. Draws for which a metric is undefined, such as AUROC with only one outcome class, follow the registered experiment’s valid-draw handling.

External patient-bootstrap intervals resample patient identifiers. Slice-level rows remain clustered within the selected patient. This avoids treating thousands of correlated slices as independent people.

## 12. Failure taxonomy

Weak segmentation is defined as Dice below 0.50. Primary and overlapping flags distinguish:

- **empty prediction:** no predicted foreground;
- **non-empty zero overlap:** foreground exists but has no reference intersection;
- **severe under-prediction:** registered area-ratio and overlap rule indicates too little coverage;
- **severe over-prediction:** registered area-ratio rule indicates excessive coverage;
- **fragmentation:** multiple predicted connected components under the registered component rule;
- **disconnected false positive:** predicted components exist away from the overlapping component; and
- **edge touch:** reference or prediction touches model-space border.

Flags may overlap. Primary mode uses the registered precedence rule so each row receives one main category. The taxonomy describes model/reference geometry; it does not identify clinical causes.

## 13. Domain-gap analysis

External studies report intensity summaries, target geometry, frozen-encoder representation distance, score distributions, and policy rates relative to matched BRISC development references. Wasserstein distance and Jensen–Shannon divergence are descriptive distribution comparisons. No single domain-gap number is interpreted as a causal explanation.

UPENN-GBM analysis uses two explicit row memberships. Mask-derived domain features use the representable-target rows. Input intensity, encoder distance, reliability, and policy analyses use all primary rows. Runtime assertions require feature-array length to equal bootstrap-membership length.

## 14. Determinism and reproducibility

Training used fixed seeds and recorded configs, but GPU training is described as controlled rather than guaranteed bitwise deterministic because the original runs did not globally enforce every deterministic-algorithm and backend setting. Inference freezing is stricter: source hashes, copied hashes, fixed thresholds, exact feature order, environment locks, synthetic fixtures, and expected output hashes are stored together.

The frozen loader verifies every allowlisted artifact and rejects missing or unexpected files. Reconstruction tolerance is 1e-12 for registered scalar/vector outputs, while hard masks and serialized hashes must match exactly. `runs/final_stack/` is immutable during Stage 9.

## 15. Interpretation rules

1. Internal validation values are labeled internal and do not imply patient-independent generalization.
2. V1 metrics are historical baselines; E08 category metrics describe the selected segmenter.
3. OOF values are reported for calibration and reliability; full-population fits are operational artifacts only.
4. External cohorts are kept separate; no pooled external performance is reported.
5. Cheng is an overlap control and cannot establish independent transport.
6. GBM-to-glioma is a coarse subtype mapping.
7. Reliability probability is not probability of clinical correctness.
8. OOD acceptance is not proof that an image is medically valid.
9. No threshold may be rounded or changed in application code without a new registered policy study.
10. All outputs retain the statement: **Research prototype — not for clinical use.**
