# Module 03 — Python & Tools Setup

> **Goal:** Set up a clean ML development environment. Learn the Python libraries (NumPy, Pandas, Matplotlib, scikit-learn) you'll use every day.

**Time:** ~6–8 hours
**Prerequisites:** Modules 01–02, basic Python

---

## 1. The Python ML Stack

Every classical ML project uses these:

| Library | What it does | When you use it |
|---------|--------------|------------------|
| **NumPy** | Fast numerical arrays | Everywhere — the foundation |
| **Pandas** | DataFrames (tables) | Loading/cleaning data |
| **Matplotlib** | Plotting | Visualizing data |
| **Seaborn** | Prettier plots | Statistical visualizations |
| **scikit-learn** | Classical ML algorithms | Training models |
| **Jupyter** | Interactive notebooks | Exploration & prototyping |

Once you move to deep learning, add **PyTorch** or **TensorFlow**.

---

## 2. Setting Up Your Environment

### Option A: Anaconda (recommended for beginners)

[Anaconda](https://www.anaconda.com/download) bundles Python + all ML libraries + Jupyter in one installer.

**Pros:** zero setup pain, just works
**Cons:** large (~3 GB), heavyweight

### Option B: Miniconda + manual installs (recommended for everyone else)

```bash
# Install Miniconda (lightweight conda)
brew install --cask miniconda    # macOS
# or download from https://docs.conda.io/en/latest/miniconda.html

# Create a fresh environment
conda create -n ml python=3.11
conda activate ml

# Install the core stack
conda install numpy pandas matplotlib seaborn scikit-learn jupyter

# Optional but nice
conda install -c conda-forge jupyterlab ipywidgets
```

### Option C: pip + venv (lightest)

```bash
python3 -m venv ~/.venv/ml
source ~/.venv/ml/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

### Verify your install

```bash
python -c "import numpy, pandas, sklearn, matplotlib; print('OK')"
jupyter lab    # opens the notebook in your browser
```

> 💡 **Tip:** Use one environment per project. Don't pollute your base install. When you finish this course, you'll have one `ml-course` environment.

---

## 3. Jupyter Notebooks

A notebook is a sequence of **cells**. Each cell is either code or markdown. You run cells one at a time and see output inline.

### Essential keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Shift + Enter` | Run cell, move to next |
| `Ctrl + Enter` | Run cell, stay |
| `Esc` then `A` | Insert cell above |
| `Esc` then `B` | Insert cell below |
| `Esc` then `D D` | Delete cell |
| `Esc` then `M` | Convert to markdown |
| `Esc` then `Y` | Convert to code |

### Your first notebook

```bash
jupyter lab
```

Create `01_hello_ml.ipynb`. Type in the first cell:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

print("Setup OK")
```

Run it (`Shift + Enter`). You should see "Setup OK".

---

## 4. NumPy Crash Course

NumPy gives you fast, vectorized arrays. Forget Python lists for ML — they're 50–100x slower.

### Creating arrays

```python
import numpy as np

# From a list
a = np.array([1, 2, 3, 4, 5])
print(a.shape)         # (5,)
print(a.dtype)         # int64

# Built-in generators
np.zeros((3, 4))       # 3x4 of zeros
np.ones((2, 2))        # 2x2 of ones
np.arange(0, 10, 2)    # [0 2 4 6 8]
np.linspace(0, 1, 5)   # [0. 0.25 0.5 0.75 1.]
np.random.randn(3, 3)  # 3x3 from normal distribution
```

### Indexing & slicing

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

A[0]          # [1 2 3]   - first row
A[:, 0]       # [1 4 7]   - first column
A[1, 2]       # 6         - row 1, col 2
A[0:2, 1:3]   # [[2 3]
              #  [5 6]]

# Boolean indexing
A[A > 5]      # [6 7 8 9]
```

### Vectorization (the magic)

NumPy operations work on entire arrays at once:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b         # [5 7 9]
a * b         # [4 10 18]   ← element-wise, NOT dot product
a @ b         # 32          ← dot product
a ** 2        # [1 4 9]
np.sqrt(a)    # [1. 1.41 1.73]
np.exp(a)     # [2.71 7.38 20.08]
```

> 🔑 **Rule:** If you ever write a `for` loop over a NumPy array, ask yourself if there's a vectorized way. There almost always is.

### Aggregations

```python
A = np.random.randn(1000, 5)

A.mean()              # overall mean
A.mean(axis=0)        # mean per column → shape (5,)
A.mean(axis=1)        # mean per row → shape (1000,)
A.sum(), A.std(), A.min(), A.max()
```

### Broadcasting (advanced but essential)

NumPy automatically expands shapes when they're compatible:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3)
b = np.array([10, 20, 30])     # shape (3,)

A + b
# → [[11, 22, 33],
#    [14, 25, 36]]
# b is "broadcast" to each row
```

This will save you tons of code. It also causes confusing bugs. Print shapes when in doubt.

---

## 5. Pandas Crash Course

Pandas gives you a **DataFrame** — basically a Python version of an Excel sheet or SQL table.

### Creating DataFrames

```python
import pandas as pd

# From a dict
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["Paris", "London", "Berlin"]
})

# From a CSV file (the most common case)
df = pd.read_csv("data.csv")

# From a URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)
```

### First things to do with a new DataFrame

```python
df.head()         # first 5 rows
df.tail(10)       # last 10 rows
df.shape          # (rows, columns)
df.columns        # column names
df.dtypes         # types per column
df.info()         # summary of types and non-null counts
df.describe()     # stats for numeric columns
df.isna().sum()   # missing values per column
```

### Selecting data

```python
df["age"]                   # one column → Series
df[["name", "age"]]         # multiple columns → DataFrame

df.iloc[0]                  # first row by integer position
df.loc[0]                   # first row by label index
df.iloc[0:5, 1:3]           # rows 0–4, columns 1–2

df[df["age"] > 28]          # boolean filter
df[(df["age"] > 28) & (df["city"] == "London")]
```

### Modifying

```python
df["age_plus_10"] = df["age"] + 10
df["upper_name"] = df["name"].str.upper()

df = df.drop(columns=["upper_name"])
df = df.rename(columns={"age": "age_years"})
```

### Group & aggregate

```python
df.groupby("city")["age"].mean()
df.groupby("city").agg({"age": ["mean", "max"], "name": "count"})
```

### Handling missing data

```python
df.dropna()                          # remove rows with any NaN
df.dropna(subset=["age"])            # remove only if 'age' is NaN
df["age"] = df["age"].fillna(df["age"].mean())   # fill with mean
df["city"] = df["city"].fillna("Unknown")        # fill with constant
```

---

## 6. Matplotlib & Seaborn

### Matplotlib — the workhorse

```python
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Sine wave")
plt.grid(True)
plt.show()
```

### Common plot types

```python
# Line plot
plt.plot(x, y)

# Scatter plot
plt.scatter(df["age"], df["income"])

# Histogram
plt.hist(df["age"], bins=20)

# Bar chart
df["city"].value_counts().plot(kind="bar")

# Multiple plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, np.sin(x))
axes[1].plot(x, np.cos(x))
```

### Seaborn — pretty defaults + statistical plots

```python
import seaborn as sns

sns.set_theme()  # nicer defaults

# Pairwise scatter for multiple columns
sns.pairplot(df, hue="species")

# Distribution of one variable
sns.histplot(df["age"], kde=True)

# Heatmap of correlations
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

# Categorical
sns.boxplot(data=df, x="city", y="age")
```

The fastest way to understand a new dataset: load it, then run `sns.pairplot(df, hue=label_col)`. Trust me.

---

## 7. scikit-learn — The Big Picture

scikit-learn is consistent. Once you learn the API, every algorithm uses the same pattern:

```python
from sklearn.SOMETHING import SomeAlgorithm

model = SomeAlgorithm(hyperparam1=..., hyperparam2=...)
model.fit(X_train, y_train)            # train
predictions = model.predict(X_test)    # predict
score = model.score(X_test, y_test)    # evaluate
```

That's it. The same 4 lines work for linear regression, random forest, SVM, neural networks. Only the imports change.

We'll use scikit-learn extensively starting in module 06. For now, just remember: `fit`, `predict`, `score`.

---

## 8. Your First Real Notebook

Create `02_first_ml_project.ipynb` and work through this:

```python
# === Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === 1. Load data ===
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

print(df.head())
print(df.describe())
print(df.shape)

# === 2. Explore ===
sns.pairplot(df, hue="species")
plt.show()

sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm")
plt.show()

# === 3. Split data ===
X = df.iloc[:, :4]      # 4 numeric features
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# === 4. Train ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === 5. Evaluate ===
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}\n")
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

Run every cell. You just:
- Loaded a dataset
- Explored it visually
- Split it for training/testing
- Trained a model
- Evaluated it with proper metrics

This is the **shape of every ML project**. The rest of the course is about making each step better.

---

## 9. Best Practices for Notebooks

- **Restart and run all** before considering anything "done." Notebooks are stateful — out-of-order execution lies.
- **Random seeds**: set `random_state=42` everywhere. Reproducibility matters.
- **Save versions**: commit your notebooks to git. Use [`nbstripout`](https://github.com/kynan/nbstripout) to strip outputs before committing.
- **One notebook per question**, not one per project. When a notebook gets long, split it.
- **Move stable code to `.py` files**. Notebooks are for exploration; modules are for production.

---

## 10. Exercises

### Exercise 1 — NumPy 50

Do exercises 1–50 from [NumPy 100 exercises](https://github.com/rougier/numpy-100). Don't peek at the answers.

### Exercise 2 — Pandas with Titanic

Load the Titanic dataset:
```python
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
```

Answer:
1. How many passengers were there?
2. What % survived?
3. What was the average age, broken down by `Sex`?
4. What was the survival rate by `Pclass`?
5. Make a histogram of ages, colored by survival.

### Exercise 3 — Visualize a dataset

Pick any dataset from [seaborn's built-in datasets](https://github.com/mwaskom/seaborn-data) or Kaggle. Make at least:
- 1 histogram
- 1 scatter plot
- 1 boxplot
- 1 correlation heatmap

Write 2–3 sentences in markdown for each, explaining what you see.

### Exercise 4 — Recreate the Iris notebook

Without copying, recreate the Iris notebook from section 8. Then change one thing (e.g., use only 2 features, or try a different model like `RandomForestClassifier`). Note what changes.

---

## 11. Key Takeaways

- **NumPy** for numbers, **Pandas** for tables, **Matplotlib/Seaborn** for plots, **scikit-learn** for models.
- The scikit-learn API is consistent: `fit`, `predict`, `score`.
- **Vectorize** everything. If you're writing a `for` loop over data, you're doing it wrong.
- Always start a new dataset with `head()`, `describe()`, `info()`, `isna().sum()`, and `pairplot()`.
- Set `random_state=42` for reproducibility.
- Notebooks for exploration; `.py` files for production.

---

## What's next?

→ **[Module 04 — Data & Features](04-Data-and-Features.md)**

Real data is messy. Module 04 teaches the unsexy but most important skill: turning raw data into features a model can learn from.

---

## References for this module

**Tutorials:**
- [NumPy: the absolute basics](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Pandas: 10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [scikit-learn: Getting started](https://scikit-learn.org/stable/getting_started.html)

**Cheat sheets:**
- [NumPy cheat sheet](https://numpy.org/devdocs/user/quickstart.html)
- [Pandas cheat sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Matplotlib cheat sheet](https://matplotlib.org/cheatsheets/)

**Practice:**
- [NumPy 100 exercises](https://github.com/rougier/numpy-100)
- [Pandas 100 exercises](https://github.com/ajcr/100-pandas-puzzles)
- [Kaggle Pandas course](https://www.kaggle.com/learn/pandas) (free, ~5 hours)
