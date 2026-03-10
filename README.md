# Spam Detection using Naive Bayes (From Scratch)

## 📌 Overview

This repository contains a from-scratch implementation of the **Naive Bayes** and **Log Naive Bayes** algorithms for binary text classification. The goal of this project is to accurately classify emails as either **Spam** or **Ham (Not Spam)** based on their word frequencies.

This project was developed as part of the curriculum for the *Mathematics for Machine Learning* specialization offered by DeepLearning.AI.

## ✨ Features

* **Built from Scratch:** No external machine learning libraries (like Scikit-Learn) were used for the core algorithms, ensuring a deep understanding of the underlying mathematics.
* **Standard Naive Bayes:** Implementation of the classic probabilistic classifier.
* **Log Naive Bayes:** A numerically stable version of the algorithm that prevents underflow errors when working with very small probabilities by using logarithmic additions.
* **Text Preprocessing:** Includes basic tokenization and word frequency counting necessary for the Bag-of-Words (BoW) model.

## 🧮 The Mathematics

The classifier is built on Bayes' Theorem. For a given email containing words $x_1, x_2, \dots, x_n$, the probability of the email belonging to class $y$ (Spam or Ham) is proportional to:

$$P(y|x_1, \dots, x_n) \propto P(y) \prod_{i=1}^n P(x_i|y)$$

### Why Log Naive Bayes?

Multiplying many small probabilities together can cause numerical underflow in standard programming languages, resulting in an output of exactly `0`. To fix this, we take the natural logarithm of the probabilities, turning the product into a sum:

$$\log P(y|x_1, \dots, x_n) \propto \log P(y) + \sum_{i=1}^n \log P(x_i|y)$$

The class that yields the highest sum is chosen as the final prediction.

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* NumPy
* Pandas 
### Installation

1. Clone the repository:
```bash
git clone https://github.com/Srihari0804/naive_bayes_and_log_naive_bayes_spam_detection.git
cd naive_bayes_and_log_naive_bayes_spam_detection

```


2. Install the required dependencies:
```bash
pip install numpy pandas

```


*(Note: Update the command above based on the actual name of your execution file).*

## 📂 Project Structure

* `Naive_bayes(1).ipynb` - Contains the implementation of the `NaiveBayes` and `LogNaiveBayes`
* `emails.csv` - emails data

## 🎓 Acknowledgments

* This project was completed as part of the **Mathematics for Machine Learning** course by DeepLearning.AI.
* Special thanks to the course instructors for breaking down the mathematical foundations of machine learning.
