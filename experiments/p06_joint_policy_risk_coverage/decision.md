# Decision — P06 joint policy risk coverage

**Status:** COMPLETE — descriptive, no gate

The exact frozen rule was applied without tuning: `ood_score < 0.36854275476100085 AND reliability >= 0.9490250173235835`.

- **brisc_validation:** STANDARD `450/680` (66.2%); accepted mean Dice `0.8894`; accepted failure rate `0.2%`; reviewed `81/82` failures and missed `1`.
- **x02_openneuro:** STANDARD `0/425` (0.0%); accepted mean Dice `NOT_ESTIMABLE_NO_STANDARD_CASES`; accepted failure rate `NOT_ESTIMABLE_NO_STANDARD_CASES`; reviewed `357/357` failures and missed `0`.
- **x03_upenn_gbm:** STANDARD `0/9464` (0.0%); accepted mean Dice `NOT_ESTIMABLE_NO_STANDARD_CASES`; accepted failure rate `NOT_ESTIMABLE_NO_STANDARD_CASES`; reviewed `7600/7600` failures and missed `0`.

These are retrospective development/external-cohort observations. They do not establish clinical safety, prospective protection, or a validated deployment policy.
