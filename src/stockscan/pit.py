"""Point-in-time (PIT) correctness primitives — the #1 invariant.

A feature row may only be used at an as-of date ``T`` if the filing it came from
was already public at ``T``. We key availability to the SEC filing date
(``sub.filed``) plus a fixed business-day ingestion lag — NOT the fiscal
period-end, which would leak weeks of look-ahead into every 10-K row.

``assert_pit`` is meant to be called inside every feature build so look-ahead is
impossible by construction rather than caught later in a suspiciously good backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AVAILABILITY_LAG_BDAYS


class LookAheadError(AssertionError):
    """Raised when a feature row would use data not yet public at the as-of date."""


def available_date(filed_date, lag_bdays: int = AVAILABILITY_LAG_BDAYS):
    """Date a filing's numbers become usable: filing date + ``lag_bdays`` business days.

    Accepts a scalar (str/date/Timestamp) or a pandas Series of datetimes and
    returns the same shape.
    """
    filed = pd.to_datetime(filed_date)
    if isinstance(filed, pd.Series):
        days = filed.values.astype("datetime64[D]")
        shifted = np.busday_offset(days, lag_bdays, roll="forward")
        return pd.Series(pd.to_datetime(shifted), index=filed.index)
    shifted = np.busday_offset(np.datetime64(filed.to_datetime64(), "D"), lag_bdays, roll="forward")
    return pd.Timestamp(shifted)


def assert_pit(df: pd.DataFrame, as_of, filed_col: str = "filed_date") -> None:
    """Fail loudly if any row's filing is not yet public at ``as_of``.

    Parameters
    ----------
    df : DataFrame that must contain ``filed_col``.
    as_of : the point-in-time snapshot date.
    filed_col : name of the filing-date column (default ``filed_date``).
    """
    if filed_col not in df.columns:
        raise KeyError(f"assert_pit: no '{filed_col}' column in frame")
    as_of = pd.Timestamp(as_of)
    avail = available_date(df[filed_col])
    violations = df.loc[avail > as_of]
    if len(violations):
        worst = pd.to_datetime(violations[filed_col]).max()
        raise LookAheadError(
            f"{len(violations)} row(s) not public at as_of={as_of.date()} "
            f"(latest filed_date={worst.date()}); look-ahead blocked."
        )
