"""Compatibility shims for Ray + pandas 3.0+.

Ray 2.54 references ``pd.errors.SettingWithCopyWarning`` which was removed in
pandas 3.0 (copy-on-write became the default behavior).  Patch it with a
stand-in so Ray's internal Arrow-to-pandas conversion doesn't crash.

This module must be imported before any Ray Data operation that triggers
pandas conversion (``map_batches(batch_format="pandas")``, ``to_pandas()``,
``iter_batches(batch_format="pandas")``).

See: https://github.com/ray-project/ray/issues/53848
"""

from __future__ import annotations

import pandas as pd

if not hasattr(pd.errors, "SettingWithCopyWarning"):
    pd.errors.SettingWithCopyWarning = FutureWarning  # type: ignore[attr-defined]
