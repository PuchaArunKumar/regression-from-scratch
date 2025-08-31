# regression-from-scratch
Implementation of types of regression from scratch
# Regression-from-Scratch

This repository implements **Regression algorithms from scratch** using only Python and its fundamental libraries (e.g., `numpy`, `pandas`). The goal is to understand the inner workings of regression models by building them step by step without relying on machine learning frameworks like scikit-learn.

---

## 📌 Features
- Implementation of **Linear Regression** (univariate and multivariate).
- Support for both **Batch Gradient Descent** and **Normal Equation** methods.
- Option to use **Stochastic Gradient Descent (SGD)** for faster training on large datasets.
- Cost function (Mean Squared Error) visualization during training.
- Comparison with scikit-learn’s regression results.

---

## 🧮 Mathematics Behind Regression

### 1. Hypothesis Function
Linear regression assumes a linear relationship between input features \(X\) and target \(y\):

\[
\hat{y} = h_\theta(X) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
\]

where:
- \(\hat{y}\) → predicted output  
- \(\theta\) → model parameters (weights)  
- \(x_i\) → input features  

---

### 2. Cost Function (Mean Squared Error)
The cost function measures how well our model fits the data:

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
\]

where:
- \(m\) → number of training examples  
- \(h_\theta(x^{(i)})\) → predicted value  
- \(y^{(i)}\) → actual value  

---

### 3. Gradient Descent Optimization
We minimize the cost function using **gradient descent**:

\[
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
\]

where:
- \(\alpha\) → learning rate  
- \(\frac{\partial J(\theta)}{\partial \theta_j}\) → gradient of cost function  

For linear regression, the gradient is:

\[
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\]

---

### 4. Normal Equation (Analytical Solution)
Instead of gradient descent, parameters can be directly computed using the normal equation:

\[
\theta = \left( X^T X \right)^{-1} X^T y
\]

where:
- \(X\) → input feature matrix (with bias term)  
- \(y\) → target vector  

This method works well for small datasets but becomes computationally expensive for very large feature sets.

---

## 📊 Results
- Trains a regression line to minimize error between predicted and actual values.
- Plots cost function reduction over epochs.
- Visualizes regression line against dataset points.

---


---

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests.

---

## 📜 License
This project is licensed under the **MIT License**.
