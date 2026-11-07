# Observation profiles experiment proposal (DEC-012 approved)

**Status: `APPROVED DECISION`.** This document proposes a controlled information-availability study; it does not change the v1 public contract. DEC-007 keeps the public `Observation` aggregate-only. The Week 5 plan itself requires a separate reviewed decision before richer disclosure is implemented.

Proposed profiles for a future, separately versioned experimental projection:

| Profile | Proposed exposure | Privacy consequence |
| --- | --- | --- |
| `aggregate_v1` | Existing aggregate support, graph summary, legal actions, budget, status, public events | Current approved baseline |
| `aggregate_plus_sample_v1` | Proposed: aggregate plus a seeded, fixed-size sample of per-voter public support bins (not exact support) | Reveals information about sampled individuals; not authorized by DEC-012 |
| `aggregate_plus_all_bins_v1` | Proposed: aggregate plus support-bin counts by declared public strata; no identifiers | Reveals distributional attributes; not authorized by DEC-012 |

DEC-012 is approved and decides to preserve aggregate-only `Observation` v1; approval of that decision is **not** approval of either proposed richer profile above. The original 25/50/75/100% labels are underspecified: they do not define fields, sampling design, identifier exposure, or repeated-round composition. No such percentages or proposed profiles are implemented or reported. A new decision must specify fields, sampling/strata, stable versus rotating samples, schema version, target-selection implications, and privacy limits before a profile can be implemented. Until then, only `aggregate_v1` experiments are valid and no information-level comparison may be claimed.
