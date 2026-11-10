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

## If I buy a party worker now, how is he or she chosen and what do they do?

Rules R21–R27, from `plans/P8_party_workers_ic.md` (2026-11-10):

- **Buying:** there are no free workers. You buy workers for a region you choose, paying a one-time fee each.
  A worker is permanent: they stay in that region for the whole game and are never reassigned.
- **Who the worker is:** a real voter in the network. The game picks a voter **at random from the region you chose**,
  among voters with **at least 3 friends** who are not already your workers. The pick uses the game's seeded
  "workers" random stream, so replaying a game reproduces it. You see how many workers you have per region, but
  not which voters they are.
- **Brainwashed:** from that moment the worker's own budget preference equals your manifesto and never changes.
  Friends, media and time do not move it. Every round they keep pulling their friends toward your manifesto
  through normal friend influence.
- **Spreading (Independent Cascade, one step per round):**
  - In the round you buy them, the worker tries once to convert each of their friends.
  - A friend who is converted takes on your manifesto, then tries **their** friends **in the next round**. The
    message moves one step through the network per round, so a worker bought in round 1 can start a cascade
    up to 5 steps deep, while one bought in the last round gets only one step.
  - After a converted voter has had their turn spreading, they are an ordinary voter again and may drift away
    under friends and media.
  - The original worker spreads only once, in the round they are bought.
- **Chance of converting a friend:** the friend's hidden receptiveness to canvassing (higher in regions whose
  personality is "Receptive to canvassing") × how likely that friend already is to vote for your manifesto.
  A friend who already likes your manifesto is easy to win over; one far from it almost never converts.
- **Limits:** each voter can be approached at most **3 times per game**, failed attempts included. Zealots and
  your own workers are never targets.
- **Why the chance includes "already likes you":** with receptiveness alone, a single worker converted about 65%
  of 10,000 voters within 5 rounds, whatever the manifesto said (each voter has about 30 friends).
