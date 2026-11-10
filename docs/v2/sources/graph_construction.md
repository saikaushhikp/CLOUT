# Multi-Agent Simulation (MAS) - Workflow & Graph Construction

## Overall Pipeline

1. **Graph Construction** $\rightarrow$ Voter
2. **Survey** $\rightarrow$
3. **Computing Result / Outcome**
4. **Media / Party Workers**
5. **Computing Result / Outcome**

---

## 1. Graph Construction

### Graph Setup & Region Splitting
- Let total voters / nodes be $x$.(typical range for this gamming will be $[1000, 100000]$,can be upto 1000000 if GPU resources are available. )
- **Regions:** Split into 4 regions (North, South, East, West: $N, S, E, W$).
  - Diagram:
    ```
          N
        \   /
      W   X   E
        /   \
          S
    ```
  - Split $x$ between 4 regions such that there is **at least $0.01 x$ in each region**.

### Community & Edge Generation
- Generate $10$ numbers:
  $$\binom{4}{1} + \binom{4}{2} = 4 + 6 = 10$$
  - $\binom{4}{1} = 4$ numbers drawn from range $[0.5, 0.9]$
  - $\binom{4}{2} = 6$ numbers drawn from range $[0.1, 0.4]$
- Use these numbers to generate edges:
  - Generate **4 strong communities** ($N, S, E, W$) using probabilities from $[0.5, 0.9]$.
  - Generate **6 weak connections between communities** ($\binom{4}{2}$) using probabilities from $[0.1, 0.4]$.

---

## 2. Voter Preference Generation

For each voter:

### Budget Preference Vector
Generate a budget preference vector across 5 dimensions / categories:
1. **Education**
2. **Freebies (Social Welfare)**
3. **Agriculture and Rural Development**
4. **Industry & Trade**
5. **Environment / Climate Action**

**Constraints:**
- $x_i \ge 0$
- $\sum x_i = 1$

### Regional Archetypes & Dirichlet Sampling
- Define **6 region archetypes**.
- For each region, randomly draw one archetype and assign it to that region.
- Each archetype has an associated mean vector.
- Use archetype mean in a **Dirichlet distribution** to generate the preference vector for each voter.

---

## 3. Survey Stage

- **Setup:**
  - Regions: North ($N$), South ($S$), East ($E$), West ($W$)
  - Region $\rightarrow$ Archetype $\rightarrow$ Budget Vector
  - Resource balance: **Money $\longleftrightarrow$ Survey**
- **Question:** What statistics are computed from the survey?

### ① Population-Level Global Statistics (Free)
From the entire population, calculate the percentage of voters having each sector in their **top 3 preferences**:
- This is available **globally** with **no money / resources spent**.
- *Example output:*
  - Agriculture: $58\%$
  - Industry: $58\%$
  - Environment: $34\%$
  - Freebies: $23\%$
  - Education: $54\%$

### ② Targeted System Queries (Resource-Gated)
The user can query the system with specific comparisons:
$$\boxed{\text{Sector 1}} \quad \boxed{\text{Sector 2}} \quad \boxed{\text{Region}}$$

- **Example Query:**
  - Sector 1: `Education`
  - Sector 2: `Agri`
  - Region: `North`
  - *Result:* $80\%$ prefer Agriculture in North.
- **Cost:**
  - Each query costs **"5" resources**.

---

## 4. Result Computation

### Distance Metric
The distance between a voter's preference vector $v$ and a budget/policy vector $b$ over the 5 sectors:
$$D_j = \frac{1}{2} \sum_{i=1}^{5} |v_i - b_i|$$

### Similarity Score
$$S_j = 1 - D_j$$

### Voting Probability
The probability of voting is modeled using a sigmoid function:
$$P(\text{vote}) = \sigma\big(\beta (S_j - z)\big)$$

where:
- $\sigma(\cdot)$ is the sigmoid activation function: $\sigma(y) = \frac{1}{1 + e^{-y}}$
- $\beta$ is a sensitivity / scaling factor
- $z$ is a threshold / baseline parameter

---

## 5. Campaigning

Campaigning mechanisms involve **Media Influence** and **Party Workers (Grassroots / Word-of-Mouth)**.

### 5.1 Media Campaigning & Opinion Dynamics
The evolution of voter $i$'s opinion/budget preference vector $x_i(t)$ at time step $t+1$ is modeled as:

$$x_i(t+1) = \big(1 - \alpha_i(t) - \eta_i\big) \, x_i(t) \;+\; \alpha_i(t) \cdot b \;+\; \eta_i \frac{\sum_{j \in \mathcal{N}_i} x_j(t)}{|\mathcal{N}_i|}$$

where:
- $b$: Campaign budget / manifesto vector.
- $\eta_i \frac{\sum_{j \in \mathcal{N}_i} x_j(t)}{|\mathcal{N}_i|}$: **Neighbor influence** from voter $i$'s local graph neighbors $\mathcal{N}_i$ with weight $\eta_i$.
- $\alpha_i(t)$: **Media influence factor**:
  - Depends on media influence / investment.
  - **No investment:** $\alpha_i(t)$ remains unchanged (e.g., becomes $0$, or default $\alpha_i \sim \mathcal{N}(0.5, 1)$).
  - **If invest:** Changes $\alpha_i(t)$ (e.g., shifts the mean of the Gaussian distribution higher).

---

### 5.2 Party Workers (Independent Cascade Model)
Party workers influence voters through an **Independent Cascade Model**:

- **Initial State:** For every bought party worker, $x_i(t) = b$.(brainwashed to the party manifesto)
- **Spread Mechanism:**
  - Party worker tries to influence adjacent neighbors based on a probability (coin flip):
    - **Heads (Success):** Neighbor is successfully influenced $\rightarrow$ converted into a temporary party worker with $x(t) = b$.
    - **Tails (Failure):** Nothing happens (fails).
- **Rounds & Cascade Rules:**
  - In the next round, the original worker **cannot influence again**.
  - Only newly converted temporary workers can propagate influence further to their neighbors.
- **Restrictions:**
  - Each voter can be targeted/attempted to be influenced at most **3 or 4 times**.
- The existing method of re-locating party members doesn't hold here. the changes to the mode shall be: once bought, he is a permanent worker (count how many you have bought). player can buy new workers in consiquent rounds and allocate them to any region. once a party worker is braniwashed, he shall stay in that region only. and the ui shall show it while playing while buying a new worker and allocating them to a region. 
- a party worker selected to a region is the node in the sample-graph(10% of that region graph) with the highest degree centrality.

### 5.3 Strategies (experiments)
1. **Strategy I: People Pleaser**
   - Choose 1 or 2 regions with a majority and invest everything (Media + Party Workers) into those regions.
2. **Strategy II: Balanced / Decision-driven**
   - Decide whether to ask queries (surveys) or not.
   - Decide whether to invest in party workers or not.
   - Decide whether to invest in media or not.
   - Choose how to craft/create the manifesto vector $b$.
3. **Strategy III: Strong-Arm / Fixed Agenda**
   - Choose a random/fixed budget and force voters to align via aggressive campaign investment.
4. **Strategy IV: Opponent-Aware**
   - Responsive strategy tailored against another player / opponent party.





