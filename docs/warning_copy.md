# User-facing reliability messages

## Usage contract

These eight complete messages implement the frozen P03 two-state reliability policy. The application selects a message by policy status, reason code, and whether the classifier-predicted category is glioma. True category never selects copy.

The policy threshold is `0.9490250173235835`. There is no Warning policy state. Every message includes the exact disclaimer **Research prototype — not for clinical use**.

Meeting the threshold means only that the output reached the registered internal reliability-policy operating point. It is not a promise of correctness or clinical validity.

## `standard_default`

- Status: `STANDARD`
- Reason: `AT_OR_ABOVE_THRESHOLD`
- Category condition: `ANY`
- Severity: `INFO`

> **Result ready for review**
>
> This output met the research prototype's internal reliability threshold of 0.9490250173235835 and can be shown in the standard view. Review the uploaded image, category label, and highlighted region before using the output.
>
> **Research prototype — not for clinical use**

## `standard_predicted_glioma`

- Status: `STANDARD`
- Reason: `AT_OR_ABOVE_THRESHOLD`
- Category condition: `PREDICTED_GLIOMA`
- Severity: `INFO`

> **Result ready for review**
>
> This output met the research prototype's internal reliability threshold of 0.9490250173235835 and can be shown in the standard view. The model-predicted category is glioma. In internal development testing, reliability ranking was weaker for glioma images. This does not change the threshold or status. Review the image, category label, and highlighted region carefully.
>
> **Research prototype — not for clinical use**

## `review_below_threshold_default`

- Status: `REVIEW_REQUIRED`
- Reason: `BELOW_THRESHOLD`
- Category condition: `ANY`
- Severity: `HIGH`

> **Manual review required**
>
> This output did not meet the research prototype's internal reliability threshold of 0.9490250173235835. Inspect the uploaded image and predicted region carefully before relying on the output.
>
> **Research prototype — not for clinical use**

## `review_below_threshold_predicted_glioma`

- Status: `REVIEW_REQUIRED`
- Reason: `BELOW_THRESHOLD`
- Category condition: `PREDICTED_GLIOMA`
- Severity: `HIGH`

> **Manual review required**
>
> This output did not meet the research prototype's internal reliability threshold of 0.9490250173235835. The model-predicted category is glioma, where reliability ranking was weaker in internal development testing. Inspect the uploaded image, category label, and predicted region carefully before relying on the output.
>
> **Research prototype — not for clinical use**

## `review_unavailable_default`

- Status: `REVIEW_REQUIRED`
- Reason: `RELIABILITY_UNAVAILABLE`
- Category condition: `ANY`
- Severity: `HIGH`

> **Manual review required**
>
> A reliability score could not be computed. The output has been placed in Review Required. The policy threshold is 0.9490250173235835, but no valid score was available for comparison. Inspect the uploaded image and predicted region carefully.
>
> **Research prototype — not for clinical use**

## `review_unavailable_predicted_glioma`

- Status: `REVIEW_REQUIRED`
- Reason: `RELIABILITY_UNAVAILABLE`
- Category condition: `PREDICTED_GLIOMA`
- Severity: `HIGH`

> **Manual review required**
>
> A reliability score could not be computed. The output has been placed in Review Required. The policy threshold is 0.9490250173235835, but no valid score was available for comparison. The model-predicted category is glioma. In internal development testing, reliability ranking was weaker for glioma images. Inspect the uploaded image, category label, and predicted region carefully.
>
> **Research prototype — not for clinical use**

## `review_invalid_default`

- Status: `REVIEW_REQUIRED`
- Reason: `RELIABILITY_INVALID`
- Category condition: `ANY`
- Severity: `HIGH`

> **Manual review required**
>
> The reliability output was invalid. The output has been placed in Review Required. A finite score between 0 and 1 was expected, so the policy threshold of 0.9490250173235835 could not be applied. Inspect the uploaded image and predicted region carefully.
>
> **Research prototype — not for clinical use**

## `review_invalid_predicted_glioma`

- Status: `REVIEW_REQUIRED`
- Reason: `RELIABILITY_INVALID`
- Category condition: `PREDICTED_GLIOMA`
- Severity: `HIGH`

> **Manual review required**
>
> The reliability output was invalid. The output has been placed in Review Required. A finite score between 0 and 1 was expected, so the policy threshold of 0.9490250173235835 could not be applied. The model-predicted category is glioma. In internal development testing, reliability ranking was weaker for glioma images. Inspect the uploaded image, category label, and predicted region carefully.
>
> **Research prototype — not for clinical use**

## Integration notes

- Display the title, body, and disclaimer together; do not omit or replace the disclaimer.
- Use the predicted-glioma variant only when the classifier output is glioma.
- Missing and invalid reliability always use `REVIEW_REQUIRED` variants.
- Keep the prediction visible in Review Required so a user can inspect it.
- Do not create a Warning state or round the stored policy threshold in machine-readable data.
