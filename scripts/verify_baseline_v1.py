"""Verify the immutable v1 baseline archive without modifying it."""

import json

from freeze_baseline_v1 import TARGET, verify_baseline

if __name__ == "__main__":
    print(json.dumps(verify_baseline(TARGET), indent=2))
