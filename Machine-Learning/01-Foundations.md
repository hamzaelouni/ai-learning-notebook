# Module 01 — Foundations

> **Goal:** Build the mental model. After this module, you'll know what ML is, what it isn't, and the vocabulary every ML practitioner uses.

**Time:** ~4–6 hours
**Prerequisites:** None

---

## 1. What is Machine Learning?

**Traditional programming:**
```
Rules + Data → Output
```
You write the rules. The computer applies them.

**Machine Learning:**
```
Data + Output → Rules
```
You give examples. The computer figures out the rules.

### A concrete example: spam detection

**Traditional approach:**
```python
def is_spam(email):
    if "viagra" in email.lower(): return True
    if "free money" in email.lower(): return True
    if email.count("!") > 5: return True
    # ... 1000 more rules ...
    return False
```
Problem: spammers change tactics. You can't keep up.

**ML approach:**
1. Collect 100,000 emails labeled "spam" or "not spam"
2. Feed them to an algorithm
3. The algorithm finds patterns: word frequencies, sender patterns, time of day, link counts...
4. The result is a **model** that predicts spam/not-spam for any new email

You never wrote a single rule. The model learned them.

### Definition

> **Machine Learning** is the field of study that gives computers the ability to learn from data without being explicitly programmed.
> — Arthur Samuel, 1959

A more modern, formal definition (Tom Mitchell, 1997):
> A computer program is said to learn from **experience E** with respect to some **task T** and some **performance measure P**, if its performance on T, as measured by P, improves with experience E.

For spam detection:
- **T** = classify emails as spam or not
- **E** = the labeled email dataset
- **P** = accuracy on a held-out test set

---

## 2. AI vs ML vs Deep Learning

```
┌──────────────────────────────────────────┐
│ Artificial Intelligence (AI)             │
│  ┌────────────────────────────────────┐  │
│  │ Machine Learning (ML)              │  │
│  │  ┌──────────────────────────────┐  │  │
│  │  │ Deep Learning (DL)           │  │  │
│  │  │  - Neural networks           │  │  │
│  │  │  - LLMs, CNNs, Transformers  │  │  │
│  │  └──────────────────────────────┘  │  │
│  │  - Decision trees, SVMs, kNN, ... │  │
│  └────────────────────────────────────┘  │
│  - Rule-based expert systems             │
│  - Search algorithms                     │
└──────────────────────────────────────────┘
```

- **AI** = any technique that makes machines appear intelligent
- **ML** = AI techniques that learn from data
- **DL** = ML techniques using deep neural networks

In this course, we focus on **classical ML** (everything in ML except deep learning). Deep learning has its own course later.

---

## 3. Types of Machine Learning

### 3.1 Supervised Learning

You give the model **labeled examples** (input + correct output). The model learns to predict the output for new inputs.
The label is only used during training to compute the error. At prediction time (in production), there is no label — that's the whole point.

Two sub-types:

| Type | Output | Example |
|------|--------|---------|
| **Regression** | A number | Predict house price ($350,000) |
| **Classification** | A category | Predict spam / not-spam |

**Examples:**
- Predicting tomorrow's temperature → regression
- Diagnosing a tumor as malignant or benign → classification (binary)
- Recognizing a digit (0–9) → classification (multi-class)
- Estimating customer lifetime value → regression

This is **80% of ML in industry**.

### 3.2 Unsupervised Learning

You give the model **unlabeled data** (just inputs, no answers). The model finds structure on its own.

| Type | Goal | Example |
|------|------|---------|
| **Clustering** | Group similar items | Customer segmentation |
| **Dimensionality reduction** | Compress data | Visualize 100D data in 2D |
| **Anomaly detection** | Find weird things | Fraud detection, broken sensors |
| **Association** | Find co-occurring items | "People who bought X also bought Y" |

**Examples:**
- Grouping news articles by topic
- Finding unusual credit card transactions
- Recommending products

### 3.3 Reinforcement Learning

The model is an **agent** in an environment. It takes actions, gets rewards or penalties, and learns to maximize total reward.

**Examples:**
- AlphaGo (game playing)
- Robotics (a robot learning to walk)
- Self-driving car decisions
- Trading bots

Not covered in detail in this course — it's a specialized field. We'll cover the basics in module 25.

### 3.4 Semi-supervised, Self-supervised

- **Semi-supervised**: small labeled dataset + large unlabeled dataset (common in real life — labels are expensive)
- **Self-supervised**: the data labels itself (e.g., predict the next word in a sentence — this is how LLMs are trained)

The typical workflow

1. Pre-train with self-supervision on massive raw data
2. Fine-tune with a small labeled dataset for your specific task

Self-supervised learning is what made the current AI boom possible — it unlocked training on the entire internet without needing anyone to label anything.

This is called pre-training + fine-tuning, and it's the foundation of modern LLMs.

---

## 4. The Core Vocabulary

You'll see these words in every ML book, paper, and code base. Memorize them.

| Term | Meaning |
|------|---------|
| **Feature** (or input, predictor, attribute, X) | A measurable property of the thing you're modeling. E.g., a house's square footage. |
| **Label** (or target, output, y) | The thing you're trying to predict. E.g., a house's price. |
| **Sample** (or instance, observation, row) | A single example. E.g., one specific house. |
| **Dataset** | A collection of samples. |
| **Model** | The mathematical object that maps features → predictions. |
| **Training** | The process of fitting the model to data. |
| **Inference** (or prediction) | Using the trained model to predict on new data. |
| **Parameter** | A value the model learns from data (e.g., the coefficients in a linear regression). |
| **Hyperparameter** | A value YOU set before training (e.g., learning rate, tree depth). |
| **Loss function** | A measure of how wrong the model is. Training minimizes this. |
| **Epoch** | One full pass through the training data. |
| **Batch** | A subset of training data used for one update step. |
| **Overfitting** | The model memorized the training data and fails on new data. |
| **Underfitting** | The model is too simple to capture the pattern. |
| **Generalization** | How well the model performs on new, unseen data. |

### A visual: the data layout

A typical ML dataset is just a table:

```
| age | income | owns_house | clicks_ad |   ← columns = features (X) + label (y)
|-----|--------|------------|-----------|
|  25 |  50000 |     0      |     1     |   ← row = sample
|  42 |  85000 |     1      |     0     |
|  31 |  62000 |     0      |     1     |
|  ...                                  |
```

- Features: `age`, `income`, `owns_house`
- Label: `clicks_ad` (binary classification)
- Each row is one sample (one user)

---

## 5. The Machine Learning Workflow

Almost every ML project follows the same 7 steps:

```
1. Define the problem
   ↓
2. Collect & explore data
   ↓
3. Clean & prepare data (preprocessing, feature engineering)
   ↓
4. Split data (train / validation / test)
   ↓
5. Choose & train a model
   ↓
6. Evaluate the model
   ↓
7. Deploy & monitor
        ↑
        └────────── (iterate)
```

### Step 1 — Define the problem

Answer these:
- Is this even an ML problem? (Sometimes a SQL query is enough.)
- Supervised or unsupervised?
- Regression or classification?
- What's the business metric? (revenue, accuracy, latency, cost...)
- What's "good enough"?

**Example:** "We want to reduce customer churn."
- ML problem: predict the probability a customer churns next month (binary classification)
- Business metric: churn rate ↓
- "Good enough": better than current heuristic (manual tagging)

### Step 2 — Collect & explore data

- Where does data come from? (logs, databases, APIs, sensors, surveys)
- Is it representative?
- How much do we have?
- **EDA (Exploratory Data Analysis):** plot histograms, correlations, missing values

### Step 3 — Clean & prepare data

- Handle missing values
- Remove duplicates
- Fix outliers
- Encode categorical variables (e.g., "Paris" → 0, "London" → 1)
- Scale numerical features
- Engineer new features (e.g., from `birth_date` → `age`)

> 🔑 **Reality check:** This step takes 60–80% of the time on real projects. Get good at it.

### Step 4 — Split the data

You need three pieces:

- **Training set (~70%)**: model learns from this
- **Validation set (~15%)**: tune hyperparameters here
- **Test set (~15%)**: final evaluation (touch only ONCE)


PHASE 1 — DEVELOPMENT (you, the data scientist, building the model)
PHASE 2 — PRODUCTION (the model is deployed, answering real customers)

The **3 sets** only exist in Phase 1. They disappear in Phase 2.

Phase 1 — Development

You have historical data: 100,000 past customers (you know who churned)

You split it:
- Training set   (70,000 customers) → model LEARNS from this
- Validation set (15,000 customers) → you COMPARE models here                                                      
- Test set       (15,000 customers) → you get FINAL HONEST SCORE here

→ You pick the best model                                                                                                                                                     
→ Phase 1 is done

Nobody is asking questions yet. No clients. Just you building and evaluating.

If you peek at the test set during development, you've leaked information and your performance estimate is optimistic.

### Step 5 — Train a model

- Pick an algorithm (linear regression? random forest? neural net?)
- Fit it on the training set
- This is usually the easiest, fastest step (`model.fit(X, y)` in scikit-learn)

### Step 6 — Evaluate

- Predict on validation/test set
- Compute metrics (accuracy, RMSE, F1, ...)
- Compare against a **baseline** (e.g., always predict the most common class)
- If it's not good enough, go back to step 3 or 5

### Step 7 — Deploy & monitor

- Wrap the model in an API
- Monitor performance — models degrade over time as data drifts
- Retrain periodically

---

## 6. A Tiny End-to-End Example

Don't worry about understanding every line. The goal is to see the workflow.

```python
# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 2. Load data
iris = load_iris(as_frame=True)
X = iris.data         # features: petal/sepal length & width
y = iris.target       # label: species (0, 1, 2)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Predict & evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
# → Accuracy: 1.00
```

That's it. **Six lines of real code, and you've trained an ML model.** We'll spend the rest of the course understanding *why* it works and *how* to do this properly.

---

## 7. When NOT to Use ML

ML is overhyped. Don't reach for it when:

- ❌ A simple rule works (`if temp > 30: alert`)
- ❌ You have very little data (< a few hundred labeled examples)
- ❌ You can't tolerate any errors (medical diagnosis with 1 sample, legal decisions)
- ❌ The patterns change daily and you can't retrain fast enough
- ❌ Stakeholders need a 100% explainable answer, and you can't use simple models

**Rule of thumb:** If a junior analyst can write the rules in a week, don't use ML.

---

## 8. Common Beginner Misconceptions

| Myth | Reality |
|------|---------|
| "ML is magic." | It's just statistics + optimization at scale. |
| "More data always wins." | Quality > quantity. Garbage in, garbage out. |
| "I need a PhD." | You need stubbornness. The math is teachable. |
| "Pick the fanciest algorithm." | Start with the simplest. Linear regression beats neural nets on small structured data. |
| "ML predicts the future." | ML extrapolates patterns. If the world changes, the model breaks. |
| "Once trained, the model is done." | Models drift. They need monitoring & retraining. |

---

## 9. Exercises

### Exercise 1 — Identify the problem type

For each scenario, identify: (a) supervised or unsupervised? (b) if supervised, regression or classification?

1. Predict the number of bikes rented tomorrow.
2. Group blog readers into "interest segments" without predefined categories.
3. Decide if a credit card transaction is fraudulent.
4. Predict the sentiment (positive/negative/neutral) of a tweet.
5. Find unusual patterns in server logs.
6. Estimate the time it will take a delivery to arrive.
7. Recommend movies similar to ones you've watched.

<details>
<summary>Answers</summary>

1. Supervised, regression
2. Unsupervised (clustering)
3. Supervised, classification (binary)
4. Supervised, classification (multi-class)
5. Unsupervised (anomaly detection)
6. Supervised, regression
7. Could be unsupervised (collaborative filtering / clustering) or supervised (predicting rating)

</details>

### Exercise 2 — Define a problem

Pick a real situation from your life or work. Write down:
- What is the goal?
- Is ML appropriate? Why or why not?
- If yes: what's the input (features), output (label), and how would you collect data?
- What's a "good enough" success metric?

### Exercise 3 — Spot the workflow

Find a Kaggle notebook (any beginner ML project) and identify the 7 workflow steps in someone else's code. Suggested starter: [Kaggle Titanic tutorial](https://www.kaggle.com/c/titanic).

---

## 10. Key Takeaways

- ML learns rules from data. Traditional programming starts with rules.
- The three big families: **supervised, unsupervised, reinforcement**.
- The vocabulary (features, labels, training, overfitting, ...) is non-negotiable — memorize it.
- Every ML project follows roughly the same 7-step workflow.
- 60–80% of real ML work is data preparation, not model training.
- ML is not magic, and it's not always the right tool.

---

## What's next?

→ **[Module 02 — Math Essentials](02-Math-Essentials.md)**

You don't need a math PhD, but you need the basics: vectors, matrices, derivatives, and probability. We'll cover exactly what you need — nothing more.

---

## References for this module

**Read:**
- *Hands-On Machine Learning* (Géron), Chapter 1 — "The Machine Learning Landscape"
- Andrew Ng's [ML Course on Coursera](https://www.coursera.org/specializations/machine-learning-introduction), Week 1
- Google's [ML Crash Course — Framing](https://developers.google.com/machine-learning/crash-course/framing/check-your-understanding)

**Watch:**
- 3Blue1Brown — [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) (preview of where we're going)
- StatQuest — [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)

**Skim:**
- The Wikipedia page for "[Machine Learning](https://en.wikipedia.org/wiki/Machine_learning)" — surprisingly good overview
