"""Consistent-world sampling, epistemic and temporal properties (P5, spec §10).

* :mod:`.knowledge` — the player's observation at an earlier decision point (``observation_at``);
* :mod:`.worlds`    — sampled information sets I(K_p) (``consistent_worlds``);
* :mod:`.epistemic` — K / M operators, policy set Π_t, outcome matrix and knowledge properties;
* :mod:`.temporal`  — bounded LTL, STL robustness, CTL-style helpers over traces / outcome trees;
* :mod:`.report`    — the post-game ``epistemic_timeline`` (web_api_v2.md, R14).
"""
