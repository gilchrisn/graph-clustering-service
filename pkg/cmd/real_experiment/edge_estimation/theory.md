## 1. Proof: The Loss of Multiplicity (The Core Flaw)

Our first proof establishes that SCAR's community aggregation method is a lossy approximation that systematically underestimates the true weighted connectivity.

### **Formal Definitions**

* **True Weighted Connectivity ($W$)**: The ground truth connectivity used in a standard weighted Louvain algorithm. It is the sum of all edge weights between two communities, $C_A$ and $C_B$.
    $$W(C_A, C_B) = \sum_{u \in C_A} \sum_{v \in C_B} |\{e(u,v) \in E\}|$$

* **Ideal Sketch-Based Connectivity ($E_{sketch}$)**: The connectivity that SCAR's model measures in an ideal state (infinite sketch size). It is the count of unique nodes in one community connected to at least one node in the other, which is what the sketch-union process measures.
    $$E_{sketch}(C_A, C_B) = |\{ v \in C_B \mid \exists u \in C_A \text{ s.t. } e(u,v) \in E \}|$$

### **The Proof**

Consider a simple graph where community $C_A = \{u_1, u_2\}$ and community $C_B = \{v_1\}$. Let two edges exist: $e(u_1, v_1)$ and $e(u_2, v_1)$.

* The **True Weighted Connectivity** is $W(C_A, C_B) = 1 + 1 = 2$.
* The **Ideal Sketch-Based Connectivity** is $E_{sketch}(C_A, C_B) = |\{v_1\}| = 1$.

Since $W > E_{sketch}$, the model is proven to be a lossy, one-sided approximation that fails to capture edge multiplicity.

***
## 2. Proof: Modeling Error vs. Estimation Error

Our second proof establishes that this flaw is a fundamental **modeling error**, not a statistical **estimation error** that can be fixed by increasing the sketch size, $K$.

### **Defining the Errors**

* **Modeling Error ($\epsilon_{model}$)**: The inherent flaw in the algorithm's design.
    $$\epsilon_{model} = |E_{sketch} - W|$$

* **Estimation Error ($\epsilon_{est}$)**: The statistical inaccuracy from using a finite sketch. Let $\hat{E}$ be the actual value SCAR computes.
    $$\epsilon_{est} = |\hat{E} - E_{sketch}|$$

### **The Proof**

Using the triangle inequality, $\epsilon_{total} \le \epsilon_{est} + \epsilon_{model}$. As the sketch size $K \to \infty$, the **estimation error** $\epsilon_{est}$ approaches zero. However, the **modeling error** $\epsilon_{model}$ is a fixed, non-zero value independent of $K$. Therefore, the total error converges to the irreducible modeling error, proving the flaw cannot be fixed by increasing sketch precision.

***
## 3. A Comparative Analysis of Connectivity Estimation Models

Given SCAR's inherent modeling error, we can compare its approach to a more direct, alternative model.

### **Model 1: Union of Member Sketches (SCAR's Approach)**

This method creates a single, aggregated sketch for a community by unioning the sketches of its members, then performs one estimation calculation.

* **Error Analysis**: This model's primary weakness is its **modeling error**. The magnitude of this error is a direct function of connection multiplicity:
    $$\epsilon_{model} = \sum_{v \in C_B, m(v, C_A) > 0} (m(v, C_A) - 1)$$
    This means the **error of this model scales with the density of inter-community connections**. In dense subgraphs or hub-and-spoke patterns, the error can become severe.

### **Model 2: Sum of Member Estimations (The Alternative Approach)**

This method estimates connectivity by summing the individual `node-to-community` estimates for each member of a community.

* **Concept**:
    $$E_{new}(C_A, C_B) = \sum_{a \in C_A} \hat{E}(a, C_B)$$
* **Error Analysis**: This model's primary weakness is its **accumulating estimation error**.
    * It has **zero modeling error**, as its ideal form is equivalent to the true weight $W$.
    * However, the statistical error from each term accumulates. The total estimation error bound is on the order of $O(|C_A|/\sqrt{K})$. This means the **error of this model scales with the number of nodes ($|C_A|$) in the source community**.

### **Conclusion: A Trade-off Between Speed and Accuracy**

The two models present a clear trade-off:

| Model | Modeling Error | Estimation Error | Speed |
| :--- | :--- | :--- | :--- |
| **1. Union of Sketches (SCAR)**| Scales with **density** | Scales with $O(1/\sqrt{K})$ | Fast (1 calculation) |
| **2. Sum of Estimations**| **Zero** | Scales with **# of nodes** ($|C_A|$) | Slow ($|C_A|$ calculations)|

The reason SCAR likely performed adequately on its test datasets (e.g., ACM, DBLP) is because academic collaboration networks are characteristically **sparse**. In a sparse graph, the inter-community connection density is naturally low, which minimizes SCAR's primary source of error. This makes its fast-but-flawed approach a viable trade-off for those specific conditions, but it would likely perform poorly on denser graphs where its modeling error would become significant.