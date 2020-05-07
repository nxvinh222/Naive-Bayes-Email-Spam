# Filter Spam Email API
## Demo: 
![api](https://user-images.githubusercontent.com/35555098/81286013-9c46e580-908a-11ea-9beb-0482d0de0a01.png)
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
- 1742 emails in test set.

## 5. Model
Using Naive Bayes:

<img src="https://render.githubusercontent.com/render/math?math=P(Spam \, | \, X) = \frac{P(X \, | \, Spam) \, P(Spam)} {P(X)}">
