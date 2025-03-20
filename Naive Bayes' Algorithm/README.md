# Naïve Bayes Algorithm

## Introduction
Naïve Bayes is a probabilistic machine learning algorithm based on Bayes' Theorem, commonly used for classification tasks. It assumes that the features are conditionally independent given the class label, making it computationally efficient and scalable for large datasets.

## History
The foundation of Naïve Bayes classification stems from the probability theory formulated by **Thomas Bayes** in the 18th century. Bayes' Theorem provided a way to update probabilities based on new evidence. In the mid-20th century, with the rise of machine learning and statistical methods, Naïve Bayes gained popularity, especially in text classification and spam filtering. Its simplicity and effectiveness in probabilistic learning have made it a widely used classification technique.

# Bayes' Theorem - The Probability Concept
Bayes' Theorem is the foundation of Naïve Bayes classification:

$$P(Y \mid X) = \frac{P(X \mid Y) P(Y)}{P(X)}$$

Where:
- $P(Y \mid X)$ is the **posterior probability** (probability of class \( Y \) given features \( X \)).
- $P(X \mid Y)$ is the **likelihood** (probability of features \( X \) given class \( Y \)).
- $P(Y)$ is the **prior probability** (probability of class \( Y \) before considering features \( X \)).
- $P(X)$ is the **evidence** (probability of features \( X \) occurring).

## Extended Bayes' Theorem
For multiple dependent variables, the Extended Bayes' Theorem is used:

$$P(Y \mid X_1, X_2, X_3) = \frac{P(X_1, X_2, X_3 \mid Y) P(Y)}{P(X_1, X_2, X_3)}$$

Using the **conditional independence assumption** in Naïve Bayes, this simplifies to:

$$P(Y \mid X_1, X_2, X_3) = \frac{P(X_1 \mid Y) P(X_2 \mid Y) P(X_3 \mid Y) P(Y)}{P(X_1, X_2, X_3)}$$

This assumption makes computations much simpler and efficient, allowing Naïve Bayes to work effectively even with large datasets.

## Law of Total Probability
The denominator $P(X)$ (evidence) in Bayes' Theorem can be computed using the Law of Total Probability:

$$P(X) = \sum_{i} P(X \mid Y_i) P(Y_i) $$

where $Y_i$ represents all possible classes. This ensures that the probabilities sum up to 1, making the classification feasible.

## Assumption of Naïve Bayes
Naïve Bayes assumes **conditional independence**, meaning:
$$P(X_1, X_2, \dots, X_n \mid Y) = P(X_1 \mid Y) P(X_2 \mid Y) ... P(X_n \mid Y)$$

This simplifies calculations but may not always hold true in real-world scenarios.


# Probability Function in Naïve Bayes (Machine Learning Algorithm)

## 1. Extended Bayes’ Theorem for Multiple Features

Given a class $Y$ and three features $X_1, X_2, X_3$, the **extended Bayes’ theorem** is:

$$P(Y \mid X_1, X_2, X_3) = \frac{P(X_1, X_2, X_3 \mid Y) P(Y)}{P(X_1, X_2, X_3)}$$


Here:
- $P(Y \mid X_1, X_2, X_3)$ → Posterior probability (probability of class \( Y \) given the features).
- $P(X_1, X_2, X_3 \mid Y)$ → Joint likelihood (probability of features given the class).
- $P(Y)$ → Prior probability of class \( Y \).
- $P(X_1, X_2, X_3)$ → Marginal probability of the features, calculated using the **Law of Total Probability**:


$$P(X_1, X_2, X_3) = \sum_{Y} P(X_1, X_2, X_3 \mid Y) P(Y)$$


## 2. Assumption of Conditional Independence (Naïve Bayes)

In Naïve Bayes, we **assume that the features are conditionally independent given the class**:

$$P(X_1, X_2, X_3 \mid Y) = P(X_1 \mid Y) P(X_2 \mid Y) P(X_3 \mid Y)$$


This simplifies the probability function:

$$P(Y \mid X_1, X_2, X_3) \propto P(Y) P(X_1 \mid Y) P(X_2 \mid Y) P(X_3 \mid Y)$$

Since the denominator $P(X_1, X_2, X_3)$ is the same for all classes, it can be ignored during classification.

## 3. Decision Rule in Naïve Bayes

The final prediction is made using the **maximum a posteriori (MAP) estimate**:

$$Y^* = \arg\max_Y P(Y) P(X_1 \mid Y) P(X_2 \mid Y) P(X_3 \mid Y)$$


This means we select the class $Y$ that **maximizes the product of the prior and the likelihoods** of the features.

---

## Summary

| **Concept** | **Formula** |
|------------|------------|
| **Extended Bayes' Theorem** | $P(Y \mid X_1, X_2, X_3) = \frac{P(X_1, X_2, X_3 \mid Y) P(Y)}{P(X_1, X_2, X_3)}$ |
| **Law of Total Probability** | $P(X_1, X_2, X_3) = \sum_Y P(X_1, X_2, X_3 \mid Y) P(Y)$ |
| **Naïve Bayes Assumption** | $P(X_1, X_2, X_3 \mid Y) = P(X_1 \mid Y) P(X_2 \mid Y) P(X_3 \mid Y)$ |
| **Final Probability Function (Naïve Bayes)** | $P(Y \mid X_1, X_2, X_3) \propto P(Y) P(X_1 \mid Y) P(X_2 \mid Y) P(X_3 \mid Y)$ |
| **Prediction Rule (MAP Estimate)** | $Y^* = \arg\max_Y P(Y) P(X_1 \mid Y) P(X_2 \mid Y) P(X_3 \mid Y)$ |

---










## Types of Naïve Bayes Classifiers and When to Use Them
1. **Gaussian Naïve Bayes** (GNB):
   - Assumes the data follows a normal (Gaussian) distribution.
   - Suitable for **continuous numerical features**.
   - Used in **medical diagnoses** and **image processing**.

2. **Multinomial Naïve Bayes** (MNB):
   - Works with discrete count-based features (e.g., word frequency in text classification).
   - Suitable for **text classification**, such as spam filtering and sentiment analysis.

3. **Bernoulli Naïve Bayes** (BNB):
   - Assumes binary-valued features (0 or 1, presence or absence of a feature).
   - Useful for **document classification** with Boolean word features.

4. **Complement Naïve Bayes** (CNB):
   - Variation of MNB but works better on **imbalanced datasets**.
   - Often used in **text classification with unbalanced classes**.

## Applications of Naïve Bayes
- **Spam filtering** (Email classification)
- **Sentiment analysis** (Classifying text as positive/negative)
- **Medical diagnosis** (Disease prediction based on symptoms)
- **Document classification** (News categorization, topic modeling)

## Advantages
- **Fast and scalable** for large datasets
- **Works well with small datasets**
- **Performs well in text classification problems**
- **Handles categorical and continuous data**

## Limitations
- **Strong independence assumption** may not hold in real-world data
- **Poor performance with highly correlated features**
- **Zero probability problem** (can be mitigated with Laplace smoothing)

## How to Use
1. **Preprocess your data** (convert categorical data if necessary)
2. **Select the appropriate Naïve Bayes variant**
3. **Train the model using training data**
4. **Make predictions on new data**
5. **Evaluate model performance using accuracy, precision, recall, etc.**

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Bayes' Theorem - Wikipedia](https://en.wikipedia.org/wiki/Bayes%27_theorem)

---
### Pritam
Will share my handwritten notes later, with more mathematical and statistical depth.
