"""Coarse sector buckets from SIC codes (SIC divisions).

Used for within-sector cross-sectional normalization. This is the standard SIC
division split (~9 buckets); a finer Fama-French 12/48 mapping is a refinement
once the full universe makes sector buckets reliably large (see DESIGN.md).
"""

from __future__ import annotations


def sic_division(sic) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if s <= 999:
        return "Agriculture"
    if s <= 1499:
        return "Mining"
    if s <= 1799:
        return "Construction"
    if s <= 3999:
        return "Manufacturing"
    if s <= 4999:
        return "Transport/Utilities"
    if s <= 5199:
        return "Wholesale"
    if s <= 5999:
        return "Retail"
    if s <= 6799:
        return "Finance"
    if s <= 8999:
        return "Services"
    return "Public/Other"
