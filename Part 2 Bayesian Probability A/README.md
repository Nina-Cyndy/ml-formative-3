# Bayesian Probability Analysis on IMDb Movie Reviews

## Project Overview

This project demonstrates the application of **Bayes' Theorem** to sentiment analysis using the IMDb Movie Reviews dataset.



The project uses Python and the Pandas library to perform probability calculations directly from the dataset.

---

# Objectives

The main objectives of this project are to:

- Understand Bayesian Probability.
- Learn how Bayes' Theorem works in sentiment analysis.
- Calculate Prior Probability.
- Calculate Likelihood.
- Calculate Marginal Probability.
- Calculate Posterior Probability.
- Practice probability computations using real-world movie review data.

---

# Dataset

The project uses the **IMDb Movie Reviews Dataset**.

The dataset contains two columns:

| Column | Description |
|---------|-------------|
| review | The text of a movie review |
| sentiment | The sentiment label (positive or negative) |

Example:

| review | sentiment |
|---------|-----------|
| This movie was excellent. | positive |
| It was boring. | negative |

---

# Technologies Used

- Python 3
- Pandas
- Jupyter Notebook

---

# Project Workflow

The notebook follows these steps.

## Step 1: Import Pandas

```python
import pandas as pd
```

Pandas is used for loading and analyzing the dataset.

---

## Step 2: Load the Dataset

```python
df = pd.read_csv("IMDB Dataset.csv")
```

The dataset is loaded into a Pandas DataFrame called `df`.

---

## Step 3: Explore the Dataset

The notebook examines the dataset using:

```python
df.head()

df.info()
```

These commands display:

- first rows
- available columns
- number of records
- data types
- missing values

---

## Step 4: Count Positive Reviews

The notebook counts how many reviews are labeled positive.

```python
positive_reviews = (df["sentiment"]=="positive").sum()
```

It also calculates the total number of reviews.

```python
total_reviews = len(df)
```

---

## Step 5: Calculate Prior Probability

Prior Probability represents the probability that a randomly selected review is positive.

Formula:


P(Positive)=\frac{\text{Positive Reviews}}{\text{Total Reviews}}


Python:

```python
prior = positive_reviews / total_reviews
```

---

## Step 6: Filter Positive Reviews

Only positive reviews are selected.

```python
positive_df = df[df["sentiment"]=="positive"]
```

This allows the project to examine only positive reviews when calculating likelihood.

---

## Step 7: Count Reviews Containing a Keyword

The keyword used is:

```python
keyword = "excellent"
```

The notebook counts how many positive reviews contain this keyword.

```python
positive_keyword_count = positive_df["review"].str.contains(
    keyword,
    case=False,
    na=False
).sum()
```

---

## Step 8: Calculate Likelihood

Likelihood measures the probability of observing the keyword given that the review is positive.

Formula:

\[
P(excellent|Positive)
=
\frac{\text{Positive Reviews containing "excellent"}}
{\text{Total Positive Reviews}}
\]

Python:

```python
likelihood = positive_keyword_count / positive_reviews
```

---

## Step 9: Calculate Marginal Probability

The notebook counts all reviews (positive and negative) containing the keyword.

```python
keyword_count = df["review"].str.contains(
    keyword,
    case=False,
    na=False
).sum()
```

Formula:

\[
P(excellent)
=
\frac{\text{Reviews containing "excellent"}}
{\text{Total Reviews}}
\]

Python:

```python
marginal = keyword_count / total_reviews
```

---

## Step 10: Calculate Posterior Probability

Finally, Bayes' Theorem is applied.

Formula:

\[
P(Positive|excellent)
=
\frac{
P(excellent|Positive)
\times
P(Positive)
}
{
P(excellent)
}
\]

Python:

```python
posterior = (likelihood * prior) / marginal
```

The result estimates the probability that a review is positive if it contains the word **"excellent"**.

---

# Understanding Bayes' Theorem

Bayes' Theorem is written as:

\[
P(A|B)
=
\frac{
P(B|A)
\times
P(A)
}
{
P(B)
}
\]

Where:

| Symbol | Meaning |
|---------|---------|
| P(A) | Prior Probability |
| P(B|A) | Likelihood |
| P(B) | Marginal Probability |
| P(A|B) | Posterior Probability |

For this project:

- A = Review is Positive
- B = Review contains "excellent"

Therefore:

\[
P(Positive|excellent)
=
\frac{
P(excellent|Positive)
\times
P(Positive)
}
{
P(excellent)
}
\]

---

# Expected Output

The notebook displays:

- Dataset information
- First rows of the dataset
- Number of positive reviews
- Total number of reviews
- Prior Probability
- Number of positive reviews containing the keyword
- Likelihood
- Number of total reviews containing the keyword
- Marginal Probability
- Posterior Probability

Example:

```
Positive Reviews: 25000

Total Reviews: 50000

Prior Probability:
0.5

Positive Reviews containing excellent:
4321

Likelihood:
0.1728

Total Reviews containing excellent:
5200

Marginal Probability:
0.104

Posterior Probability:
0.8308
```

*(Actual values depend on the dataset.)*

---

# Mathematical Summary

Prior:

\[
P(Positive)
=
\frac{Positive}{Total}
\]

Likelihood:

\[
P(excellent|Positive)
=
\frac{Positive\ Reviews\ with\ excellent}
{Positive\ Reviews}
\]

Marginal:

\[
P(excellent)
=
\frac{Reviews\ with\ excellent}
{Total\ Reviews}
\]

Posterior:

\[
P(Positive|excellent)
=
\frac{
P(excellent|Positive)
\times
P(Positive)
}
{
P(excellent)
}
\]

---

# Folder Structure

```
project/
│
├── Bayesian_Probability.ipynb
├── IMDB Dataset.csv
└── README.md
```

---

# How to Run

1. Install Python 3.
2. Install Pandas.

```bash
pip install pandas
```

3. Download the IMDb Dataset.
4. Place the dataset in the same directory as the notebook.
5. Open the notebook using Jupyter Notebook or Google Colab.
6. Run each cell in order.

---

# Learning Outcomes

After completing this project, you will understand:

- Bayesian Probability
- Bayes' Theorem
- Prior Probability
- Likelihood
- Marginal Probability
- Posterior Probability
- Text filtering using Pandas
- Basic sentiment analysis concepts
- Probability calculations using real-world datasets

---

# Future Improvements

Possible enhancements include:

- Using multiple keywords instead of one.
- Implementing a full Naïve Bayes classifier.
- Tokenizing review text.
- Removing stop words.
- Applying Laplace Smoothing.
- Comparing results with Scikit-learn's `MultinomialNB`.
- Evaluating model performance using accuracy, precision, recall, and F1-score.

---

# Author

SEMANA SHEMA Parfait
