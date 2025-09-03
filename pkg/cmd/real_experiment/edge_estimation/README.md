# Controlled Experiment: Sketch-Based Community Connectivity Estimation

## 1. Objective

This project provides a controlled experiment (implemented in Go) to analyze and validate the theoretical limitations of the community connectivity estimation method used in the SCAR algorithm, described in *“Community Detection in Heterogeneous Information Networks Without Materialization.”*

The experiment compares two approaches for estimating inter-community edge weights using Bottom-K sketches:

* **Model 1 — Union of Member Sketches (SCAR’s Method):**
  Aggregates a community sketch by taking the union of all member sketches.
  *Pros:* very fast.
  *Cons:* hypothesized to have a fundamental **modeling error**.

* **Model 2 — Sum of Member Estimations (Alternative Method):**
  Computes edge weights by summing individual node-to-community estimates.
  *Pros:* zero modeling error.
  *Cons:* estimation error may accumulate with community size.

---

## 2. Background & Hypotheses

SCAR defines a community sketch as:

$$
vbks(C) = \bigcup_{u \in C} vbks(u)
$$

This reduces connectivity information to a set of unique neighbors and ignores edge multiplicity, systematically **underestimating true edge weights**.

We test three hypotheses:

* **H1 (Density Scaling):** Error of SCAR’s union method grows with the density/multiplicity of inter-community connections.
* **H2 (Size Scaling):** Error of the sum method grows with the number of nodes in the source community (accumulated estimation error).
* **H3 (Sparse Graphs):** SCAR’s union method performs adequately in sparse settings (e.g. planted partitions) where inter-community density is low.

---

## 3. Formal Analysis

### Proof 1: Loss of Multiplicity

* **True weighted connectivity:**

$$
W(C_A, C_B) = \sum_{u \in C_A} \sum_{v \in C_B} |e(u,v)|
$$

* **SCAR sketch connectivity (ideal, infinite $k$):**

$$
E_{sketch}(C_A, C_B) = |\{ v \in C_B \mid \exists u \in C_A : e(u,v) \in E \}|
$$

**Counterexample:**
If $C_A = \{u_1,u_2\}, C_B=\{v_1\}$, with edges $(u_1,v_1), (u_2,v_1)$:

* $W=2$
* $E_{sketch}=1$

Thus, SCAR underestimates multiplicity → **lossy approximation**.

---

### Proof 2: Modeling Error vs Estimation Error

* **Modeling error (irreducible):**

$$
\epsilon_{model} = |E_{sketch} - W|
$$

* **Estimation error (finite sketch size):**

$$
\epsilon_{est} = |\hat{E} - E_{sketch}|
$$

By triangle inequality:

$$
\epsilon_{total} \le \epsilon_{model} + \epsilon_{est}
$$

As $k \to \infty$, $\epsilon_{est} \to 0$.
But $\epsilon_{model} > 0$ whenever edge multiplicity exists.
👉 Total error never vanishes: **irreducible modeling error**.

---

### Error Characterization

* **Union of Sketches (SCAR):**

  * *Modeling error:* scales with density / multiplicity.
  * *Estimation error:* scales as $O(1/\sqrt{k})$.
  * *Speed:* fast (1 calculation).

* **Sum of Estimations (Alternative):**

  * *Modeling error:* zero.
  * *Estimation error:* scales as $O(|C_A|/\sqrt{k})$.
  * *Speed:* slower (linear in community size).

---

## 4. Experiment Setup

### Prerequisites

* Go 1.18+.

### Run

```bash
go run main.go
```

The program runs all pre-configured experiments and produces two outputs.

### Configuration

Experiments are defined in `createExperiments()` in `main.go`.
You can adjust:

* `NumNodes`: number of nodes.
* `EdgeProb`: edge probability (Erdős–Rényi graphs).
* `K`: sketch size.
* `Repetitions`: number of random seeds per config.

Pre-configured test series:

* **Density Series:** fixed $N$, varying $p$.
* **Size Series:** fixed $p$, varying $N$.
* **Planted Partition Series:** sparse graphs with known structure.

---

## 5. Output

Two files are generated:

1. **`controlled_experiment_results.json`**
   Machine-readable raw data (for custom analysis/plots).

2. **`controlled_experiment_report.txt`**
   Human-readable summary validating the hypotheses.

### Report Sections

* **Test 1: Union Error vs Density** → confirms H1.
* **Test 2: Sum Error vs Size** → confirms H2.
* **Test 3: Sparse Graph Analysis** → confirms H3.
* **Overall Conclusions:** ✓ or ✗ for each hypothesis.

---

## 6. Code Structure

All logic is in `main.go`, organized as:

* **Data Structures:** Graph, BottomKSketch, Result structs.
* **Experiment Config:** parameter definitions.
* **Graph Generation:** Erdős–Rényi + Planted Partition.
* **Sketch Operations:** bottom-k implementation.
* **Estimation Methods:** union vs sum.
* **Ground Truth:** exact edge weights.
* **Execution Loop:** runs experiments.
* **Analysis:** processes results + generates reports.

No external dependencies; uses only the Go standard library.

---

## 7. Why SCAR Looked Good in Original Paper

Academic collaboration networks (SCAR’s test case) are sparse by nature.

* Sparse graphs = low inter-community density = minimal modeling error.
* Thus SCAR’s speed advantage outweighs its approximation flaw in that domain.

But in **dense graphs or graphs with edge multiplicity**, the modeling error dominates and becomes clearly visible.
