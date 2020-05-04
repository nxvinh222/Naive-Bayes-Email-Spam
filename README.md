# Filter Spam Email using Naive Bayes

## 1. Problem

Identifying Spam Email given an Email.


## 2. Data

https://spamassassin.apache.org/old/publiccorpus/

## 3. Evaluation

Classify emails in test set and compare with their true labels to get accuracy.

## 4. Features

About the data:
- Labeled Data (spam email or not).
- 5799 emails (both spam and non-spam)
- 4057 emails in training set.
- 1742 images in test set.

## 5. Model
Using Naive Bayes:

<img src="https://render.githubusercontent.com/render/math?math=P(Spam \, | \, X) = \frac{P(X \, | \, Spam) \, P(Spam)} {P(X)}">
