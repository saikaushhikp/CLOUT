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

Campaigning mechanisms involve:
- **Party Workers** (fixed in number)
- **Media** (single media)
- **Ideation** (still in development)


