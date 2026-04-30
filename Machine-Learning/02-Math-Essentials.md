# Module 02 — Math Essentials for ML

> **Goal:** Learn the *minimum* math that lets you read ML papers, debug models, and understand error messages. We're skipping anything you don't need in the first year.

**Time:** ~6–10 hours
**Prerequisites:** Module 01

---

## How to use this module

You don't need to memorize every formula. You need to:
1. Recognize each concept when you see it
2. Know what it does intuitively
3. Know how to compute it in NumPy

If you forget a formula, that's fine. If you can't tell a vector from a matrix, that's not fine.

---

## 1. Linear Algebra

ML is built on linear algebra. Every dataset is a matrix, every prediction is a dot product, every neural network is a chain of matrix multiplications.

### 1.1 Scalars, Vectors, Matrices, Tensors

| Object | Shape | Example | NumPy |
|--------|-------|---------|-------|
| **Scalar** | () | `3.14` | `np.array(3.14)` |
| **Vector** | (n,) | a list of 5 numbers | `np.array([1,2,3,4,5])` |
| **Matrix** | (m, n) | a table | `np.array([[1,2],[3,4]])` |
| **Tensor** | (a, b, c, …) | n-dimensional array | `np.zeros((2,3,4))` |

In ML:
- A **single sample** is a vector
- A **dataset** is a matrix (rows = samples, columns = features)
- An **image** is a tensor (height × width × color channels)

### 1.2 Vector Operations

Given `a = [1, 2, 3]` and `b = [4, 5, 6]`:

**Addition** (element-wise):
```
a + b = [5, 7, 9]
```

**Scalar multiplication:**
```
2 * a = [2, 4, 6]
```

**Dot product** (sum of element-wise products):
```
a · b = 1*4 + 2*5 + 3*6 = 32
```

The dot product is the single most important operation in ML. Almost every model computes it. In linear regression, the prediction is literally `weights · features`.

```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)   # → 32
a @ b          # → 32 (the @ operator does matrix multiplication)
```

### 1.3 Matrix Multiplication

If A is (m × n) and B is (n × p), then `A @ B` is (m × p).

**The rule:** the inner dimensions must match.

```
A: (2, 3)    B: (3, 4)    A @ B: (2, 4)  ✓
A: (2, 3)    B: (4, 3)    A @ B: ERROR   ✗
```

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3)
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])          # shape (3, 2)
A @ B
# → [[ 4,  5],
#    [10, 11]]
# shape (2, 2)
```

> 🔑 **The #1 source of bugs in ML code is shape mismatches.** Print `.shape` constantly. Get a feel for which dimension is "samples" and which is "features".

### 1.4 Transpose

Flip rows and columns. Notation: `A^T` or `A.T`.

```
A = [[1, 2, 3],     A.T = [[1, 4],
     [4, 5, 6]]            [2, 5],
                           [3, 6]]
```

### 1.5 Norms (length of a vector)

The **L2 norm** (Euclidean length):
```
||v||₂ = √(v₁² + v₂² + ... + vₙ²)
```

For `v = [3, 4]`: `||v|| = √(9+16) = 5`. (The classic 3-4-5 triangle.)

The **L1 norm** (Manhattan distance):
```
||v||₁ = |v₁| + |v₂| + ... + |vₙ|
```

Norms matter for **regularization** (module 14) — we'll penalize models with large weights.

```python
np.linalg.norm([3, 4])        # → 5.0  (L2 by default)
np.linalg.norm([3, 4], ord=1) # → 7.0  (L1)
```

### 1.6 What you can skip (for now)

- Eigenvalues / eigenvectors → needed only for PCA (module 18)
- Matrix inverse → almost never computed by hand in ML
- Determinants → rarely used

---

## 2. Calculus

ML is **optimization**: find the parameters that minimize the loss. To minimize a function, you need derivatives.

### 2.1 Derivative — The slope

The derivative of `f` at point `x` is the slope of the tangent line at that point. Notation: `f'(x)` or `df/dx`.

Intuition: "If I nudge `x` a tiny bit, how much does `f(x)` change?"

| Function | Derivative |
|----------|-----------|
| `f(x) = c` (constant) | `0` |
| `f(x) = x` | `1` |
| `f(x) = x²` | `2x` |
| `f(x) = xⁿ` | `n · xⁿ⁻¹` |
| `f(x) = e^x` | `e^x` |
| `f(x) = ln(x)` | `1/x` |

These are the only ones you need to memorize.

### 2.2 The Chain Rule

If `y = f(g(x))`, then:
```
dy/dx = f'(g(x)) · g'(x)
```

Example: `y = (3x + 1)²`
- Let `g(x) = 3x + 1`, so `g'(x) = 3`
- Let `f(u) = u²`, so `f'(u) = 2u`
- Then `dy/dx = 2(3x+1) · 3 = 6(3x+1)`

The chain rule is the engine behind **backpropagation** (module 25). Don't worry — you'll use libraries that compute gradients automatically (autograd, PyTorch, TensorFlow). But you should know what they're doing.

### 2.3 Partial Derivatives

When `f` has multiple inputs (e.g., `f(x, y) = x² + 3xy`), the partial derivative with respect to `x` treats `y` as a constant:
```
∂f/∂x = 2x + 3y
∂f/∂y = 3x
```

### 2.4 The Gradient

The **gradient** ∇f is the vector of all partial derivatives:
```
∇f = [ ∂f/∂x, ∂f/∂y, ∂f/∂z, ... ]
```

Geometrically: the gradient points in the direction of steepest ascent.

**Gradient descent** — the most important algorithm in ML — does the opposite: it follows `-∇f` (steepest descent) to find the minimum.

```
repeat:
    θ_new = θ_old - α · ∇L(θ_old)
```

- `θ` (theta) = the model's parameters
- `L` = loss function
- `α` (alpha) = learning rate (a hyperparameter)

We'll see this in action in module 06.

```python
# Numerical gradient — just for intuition (real ML uses autograd)
def f(x):
    return x**2 - 4*x + 5

def gradient_descent(start=0, lr=0.1, steps=20):
    x = start
    for _ in range(steps):
        grad = 2*x - 4   # derivative of f
        x = x - lr * grad
    return x

gradient_descent()  # → ~2.0 (the minimum of f)
```

---

## 3. Probability & Statistics

Models output probabilities. Metrics are statistical estimates. You need this.

### 3.1 Random Variables, Distributions

A **random variable** is a variable whose value depends on chance. Examples:
- `X` = result of a coin flip (0 or 1)
- `X` = height of a random person

A **probability distribution** describes how likely each value of X is.

**Two main types:**
- **Discrete** (countable values): coin flip, die roll
- **Continuous** (uncountably many values): height, weight, temperature

### 3.2 Key Distributions

**Bernoulli** — one coin flip with probability p of success.
- `P(X=1) = p, P(X=0) = 1-p`
- Used in: binary classification (model outputs probability of class 1)

**Binomial** — n independent Bernoulli trials.
- "How many heads in 10 flips?"

**Normal (Gaussian)** — the bell curve.
- Defined by mean μ and standard deviation σ
- Tons of natural phenomena are roughly normal
- Notation: `X ~ N(μ, σ²)`
- The Central Limit Theorem says: averages of many random things tend to be normal

**Uniform** — every value equally likely.

You don't need to derive these. You just need to recognize them.

### 3.3 Mean, Variance, Standard Deviation

For a dataset `[x₁, x₂, ..., xₙ]`:

**Mean (average):**
```
μ = (x₁ + x₂ + ... + xₙ) / n
```

**Variance** (average squared distance from the mean):
```
σ² = ( (x₁-μ)² + (x₂-μ)² + ... + (xₙ-μ)² ) / n
```

**Standard deviation:** `σ = √(σ²)`

Variance and std dev measure **spread**. High variance = data is scattered.

```python
data = [1, 2, 3, 4, 5]
np.mean(data)   # → 3.0
np.var(data)    # → 2.0
np.std(data)    # → 1.41
```

### 3.4 Conditional Probability

`P(A | B)` = "probability of A given B happened."
```
P(A | B) = P(A and B) / P(B)
```

Example: P(spam | email contains "free") might be 80%.

### 3.5 Bayes' Theorem

```
P(A | B) = P(B | A) · P(A) / P(B)
```

This is the foundation of **Naive Bayes** classifier (module 10). It also shows up everywhere in modern ML (Bayesian methods).

Read it as: "given some new evidence B, here's how to update your belief about A."

### 3.6 Independence & Correlation

Two variables are **independent** if knowing one tells you nothing about the other.

**Correlation** measures linear relationship between two variables. Range: -1 to +1.
- `+1`: perfect positive linear relationship
- `0`: no linear relationship
- `-1`: perfect negative linear relationship

⚠️ **Correlation ≠ causation.** Ice cream sales and drowning deaths are correlated. (Both are caused by hot weather.)

```python
np.corrcoef([1,2,3,4], [2,4,6,8])
# → 1.0   (perfect correlation)
```

### 3.7 Sampling, Bias, the Law of Large Numbers

- **Sample**: a subset drawn from a population
- **Sampling bias**: when your sample doesn't represent the population (this *will* break your model)
- **Law of Large Numbers**: as your sample size grows, the sample mean approaches the true mean

In ML, biased samples are a top cause of unfair models. Always ask: is my data representative?

---

## 4. Information Theory (just enough)

You'll see these terms in classification:

**Entropy** — uncertainty in a distribution.
```
H(X) = -Σ P(x) · log P(x)
```
- A fair coin has high entropy (you can't predict it)
- A loaded coin (99% heads) has low entropy

**Cross-entropy** — measures how different two distributions are. Used as a **loss function** for classification:
```
H(p, q) = -Σ p(x) · log q(x)
```
where `p` is the true distribution and `q` is the model's predicted distribution.

You don't need to derive this. You need to recognize "cross-entropy loss" when you see it.

---

## 5. Putting It Together — A Linear Regression by Hand

Let's combine everything. Linear regression predicts `y = w · x + b`.

Given data `(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)`, we want to find `w` and `b` minimizing:
```
L(w, b) = (1/n) Σ (y_i - (w·x_i + b))²    ← Mean Squared Error
```

The gradients (using calculus):
```
∂L/∂w = -(2/n) Σ x_i · (y_i - (w·x_i + b))
∂L/∂b = -(2/n) Σ (y_i - (w·x_i + b))
```

Gradient descent update:
```
w := w - α · ∂L/∂w
b := b - α · ∂L/∂b
```

That's it. **That's a whole ML algorithm.** No magic — just linear algebra (the dot product), calculus (the gradient), and statistics (the mean).

```python
import numpy as np

# Generate fake data: y = 2x + 1 + noise
np.random.seed(42)
X = np.random.randn(100)
y = 2 * X + 1 + 0.1 * np.random.randn(100)

# Train by hand
w, b = 0.0, 0.0
lr = 0.1

for step in range(1000):
    y_pred = w * X + b
    error = y - y_pred
    grad_w = -2 * np.mean(X * error)
    grad_b = -2 * np.mean(error)
    w -= lr * grad_w
    b -= lr * grad_b

print(f"w ≈ {w:.3f} (true: 2.0)")
print(f"b ≈ {b:.3f} (true: 1.0)")
```

Run this. **You just trained a model from scratch using nothing but arithmetic.** Everything else in this course is generalization of this idea.

---

## 6. Exercises

### Exercise 1 — NumPy warmup

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([1, 0])
```

Compute by hand, then check with NumPy:
1. `A + B`
2. `A @ B`
3. `A @ v`
4. `A.T`
5. `np.linalg.norm(v)`

### Exercise 2 — Derivatives

Compute `df/dx`:
1. `f(x) = 5x³ - 2x + 7`
2. `f(x) = (2x + 1)³`  *(use chain rule)*
3. `f(x, y) = x²y + 3xy²`  *(both partials)*

<details><summary>Answers</summary>

1. `15x² - 2`
2. `6(2x+1)²`
3. `∂f/∂x = 2xy + 3y²` ;  `∂f/∂y = x² + 6xy`

</details>

### Exercise 3 — Probability

A test for a disease is 99% accurate. The disease affects 1 in 1000 people. You test positive. What's the probability you have the disease?

(Hint: Bayes' theorem.)

<details><summary>Answer</summary>

About 9%. (This is the classic "base rate fallacy" problem — most positives are false positives because the disease is rare.)

</details>

### Exercise 4 — Code the gradient descent above

Run the linear regression code in section 5. Then:
- Change the learning rate to 1.0. What happens?
- Change it to 0.001. What happens?
- Add more noise. Does it still work?

This builds intuition for how learning rate matters.

---

## 7. Key Takeaways

- **Linear algebra**: vectors, matrices, dot products, matrix multiplication, transpose, norms.
- **Calculus**: derivatives, chain rule, partial derivatives, gradients. Gradient descent = follow the negative gradient.
- **Probability/Stats**: distributions, mean/variance, conditional probability, Bayes' theorem, correlation.
- **Information theory**: entropy, cross-entropy (the most common classification loss).
- Print `.shape` constantly. Shape mismatches cause more bugs than anything else.
- Libraries do the math for you. You need to *understand* it, not *do* it.

---

## What's next?

→ **[Module 03 — Python & Tools Setup](03-Python-Setup.md)**

We'll set up the actual environment: Python, conda, NumPy, Pandas, scikit-learn, Jupyter. Then you'll write your first real ML notebook.

---

## References for this module

**Read:**
- *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) — free online at https://mml-book.github.io/
- *Hands-On Machine Learning* (Géron), Appendix A — Linear Algebra refresher

**Watch (highly recommended):**
- 3Blue1Brown — [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (15 short videos — the best linear algebra resource on the internet)
- 3Blue1Brown — [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- StatQuest — [Statistics Fundamentals playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)

**Practice:**
- Khan Academy — [Linear Algebra](https://www.khanacademy.org/math/linear-algebra) and [Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)
- [NumPy 100 exercises](https://github.com/rougier/numpy-100) — best way to drill NumPy
