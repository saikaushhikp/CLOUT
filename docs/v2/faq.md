# Dumbocracy v2 — Player FAQ

Answers to questions from the team (2026-11-09). Rule references point to `technical_spec_v2.md` and
`decisions_v2.md`.

## What does "Blend my hunches" do, and how does it work?

In the survey desk you can pencil in a **hunch** for each region: which of the six public archetypes you think
lives there (for example "North looks like Green urban"). The region dossiers help by ranking the archetypes that
best explain your survey answers, weighting each answer by its margin of error.

**Blend my hunches** turns those hunches into a manifesto:

1. You choose **which regions to blend**: at least 2, up to all 4. Shortcuts pick the **2 largest** or
   **3 largest** regions. A region with no hunch can't be selected. (R20)
2. For each chosen region it takes your hunched archetype's **average budget profile** μ, the public mean split
   over Education, Welfare, Agriculture, Industry and Environment.
3. It averages those profiles, **weighted by how many voters live in each region** (public region sizes):

   b = Σ_r n_r · μ_hunch(r) / Σ_r n_r over the chosen regions r,

   then rounds to whole percentages that sum to exactly 100.
4. The sliders are set to that split. You can still adjust them before you commit.

Why blend only 2 or 3 regions? You win with more than half the votes, and voters back a manifesto only when it is
close to their own budget (similarity above about 82.5%; spec §5). A manifesto that averages all four regions can
end up close to nobody. Aiming at a coalition of 2–3 regions that together hold over 50% of voters, and whose
archetypes are compatible, is often stronger. Remember that friends pull voters toward their regional mean during
the campaign, so aim at the electorate as it will look on election day.

Blending is a starting point, not an optimiser: it does not use your poll results, personalities or the network.

## If I hire a party worker now, how is he or she chosen and used?

- **Hiring (R16):** you pay a one-time fee of ₫4.00 per worker in the round you hire. The worker then joins your
  permanent team: they work in every remaining round, exactly like your 10 free workers, at no further cost.
- **Who the worker is:** workers are not individual simulated people. A worker is a unit of canvassing capacity.
  Nothing about a hired worker is drawn at random; all workers are identical.
- **Assignment:** each round you assign every worker on your team (free + hired) to a region. A worker you leave
  unassigned idles that round.
- **Who they canvass:** each assigned worker contacts `q = max(1, round(0.25% × N))` voters, drawn uniformly at
  random **with replacement** from that region's voters (25 per worker at N = 10,000). The draw uses the game's
  seeded "workers" random stream, so replaying a game reproduces it. A voter contacted twice gets two pulls.
- **Effect on a contacted voter:** each contact pulls the voter's budget toward your manifesto by `0.5 × γ_i`,
  where γ_i is that voter's (hidden) susceptibility to canvassing. It is higher in regions whose hidden
  personality is "Receptive to canvassing", and zero for zealots. Several contacts compound: 1 − (1 − 0.5 γ_i)^k.
- **Word of mouth:** every contacted non-zealot then spreads the message along friendship ties for up to 3 hops,
  reaching each friend with probability 0.03 per hop; spreaders stop with probability 0.5 per hop (an SIR
  cascade). Each voter reached this way gets a smaller pull, `0.15 × γ_j`. Workers therefore pay off most in dense,
  receptive regions, where the cascade is above its spectral threshold (experiment E3).

Values are the calibrated defaults (spec §14) and may change after the next round of mechanics changes.
