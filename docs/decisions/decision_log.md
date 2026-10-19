# Dumbocracy Decision Register

On 2026-10-19 the project owner reported approval of every Week 1 blocker by
all five team members. Names were not supplied and are deliberately not
fabricated here.

| ID | Topic | Decision | Status | Approved by | Date |
| --- | --- | --- | --- | --- | --- |
| DEC-001 | Runtime/test policy | Python; CPU correctness; optional CUDA only after profiling | Approved constraint | Project owner/team | Ongoing |
| DEC-002 | Player/campaign | One controllable campaign; competitors deferred | Approved | All five team members | 2026-10-19 |
| DEC-003 | Voter semantics | Scalar `support` in [0,1]; terminal vote derived; turnout deferred | Approved | All five team members | 2026-10-19 |
| DEC-004 | Topology | Fixed undirected unweighted graph; ER and edge-list generators | Approved | All five team members | 2026-10-19 |
| DEC-005 | Dynamics | Synchronous bounded neighbor-plus-campaign update in Technical Spec §4 | Approved | All five team members | 2026-10-19 |
| DEC-006 | Initial actions | Deterministic budgeted targeted contact and broadcast | Approved | All five team members | 2026-10-19 |
| DEC-007 | Information model | Private true state; public aggregate/metadata/actions/budget | Approved | All five team members | 2026-10-19 |
| DEC-008 | Horizon/result | Fixed horizon; terminal vote share strictly over 0.5 wins | Approved | All five team members | 2026-10-19 |
| DEC-009 | Logic semantics | Bounded-LTL completed traces; same-round observation equality | Approved | All five team members | 2026-10-19 |
| DEC-010 | Reproducibility | Root seed by subsystem; capture config/revision/environment/backend | Approved constraint | Project owner/team | Ongoing |
| DEC-011 | M0 review process | Five individual reviews required before freeze | Approved | All five team members | 2026-10-19 |

## Frozen contract rule

DEC-002 through DEC-009 are frozen M0 requirements. A later change requires a
new decision record and compatibility review. Week 2 may implement them without
guessing.
