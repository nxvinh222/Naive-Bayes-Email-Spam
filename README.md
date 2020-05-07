# Filter Spam Email API
## Demo: 
![api](https://user-images.githubusercontent.com/35555098/81286013-9c46e580-908a-11ea-9beb-0482d0de0a01.png)
## 1. Problem

Identifying Spam Email given an Email.


## 2. Data

https://spamassassin.apache.org/old/publiccorpus/


## 3. Features

About the data:
- Labeled Data (spam email or not).
- 5799 emails (both spam and non-spam)

## 4. Model
Using Naive Bayes:

<img src="https://render.githubusercontent.com/render/math?math=P(Spam \, | \, X) = \frac{P(X \, | \, Spam) \, P(Spam)} {P(X)}">

## 5. Install
- Download Data above
- Configure path yourself
- Run 3 notebook by order: 
  + Bayes Classifier Pre-Processing.ipynb
  + Bayes Classifier - Training.ipynb
  + Bayes Classifier - Testing, Inference & Evaluation.ipynb
- Run server.py
