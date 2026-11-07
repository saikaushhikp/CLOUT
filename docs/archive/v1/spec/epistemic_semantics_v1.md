# Finite-world epistemic semantics v1

These are analysis semantics, not engine terminal rules. A world is a complete server-side `GameState` supplied to an analysis call. The checker may inspect it only to execute actions through `GameEngine.step`; player equivalence is defined by the allowlisted public `Observation` plus public history.

- `PossibleVictory`: some supplied world has a legal action sequence ending in the official fixed-horizon result `victory == true`.
- `GuaranteedVictory`: one action policy, selecting identical actions in indistinguishable worlds, reaches victory in every supplied world.
- `InevitableDefeat`: no enumerated legal action sequence wins in any supplied world (the negation of possible victory over this finite abstraction).
- `K_player(P)`: P holds in every world in the current observation-equivalence class. It is not inferred from aggregate means alone. For this implementation, `K_player(GuaranteedVictory)` is reported with the same value as `GuaranteedVictory` for each supplied observation class (the policy may condition on the public observation); `K_player(InevitableDefeat)` matches `InevitableDefeat` across that class.

The v1 implementation abstracts targeted contacts to the first eight lexicographically ordered public singleton IDs, alongside broadcast and no-op. Default caps are 10,000 total search nodes across possible/guaranteed checks, 128 rounds, and 256 input worlds; larger worlds/horizons and node-cap exhaustion return `INCOMPLETE`, never a proof. Consequently “exact” means exact for that named finite set and action abstraction only. No checker result terminates an episode early. World construction/representativeness is the caller's responsibility; continuous hidden-state universality is not claimed.