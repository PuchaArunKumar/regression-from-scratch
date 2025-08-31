# Regression-from-Scratch
This repository implements **Regression algorithms from scratch** using only Python and its fundamental libraries (e.g., `numpy`, `pandas`). The goal is to understand the inner workings of regression models by building them step by step without relying on machine learning frameworks like scikit-learn.

## 📌 Features
- Implementation of **Linear Regression** (univariate and multivariate).
- Support for both **Batch Gradient Descent** and **Normal Equation** methods.
- Option to use **Stochastic Gradient Descent (SGD)** for faster training on large datasets.
- Cost function (Mean Squared Error) visualization during training.
- Comparison with scikit-learn’s regression results.

## 🧮 Mathematics Behind Regression
### 1. Hypothesis Function
Linear regression assumes a linear relationship between input features **X** and target **y**:
ŷ = hθ(X) = θ0 + θ1x1 + θ2x2 + ... + θnxn
where:
- ŷ → predicted output  
- θ → model parameters (weights)  
- xi → input features  

### 2. Cost Function (Mean Squared Error)
The cost function measures how well our model fits the data:
J(θ) = (1 / 2m) Σ ( hθ(xᶦ) - yᶦ )²
where:
- m → number of training examples  
- hθ(xᶦ) → predicted value  
- yᶦ → actual value  

### 3. Gradient Descent Optimization
We minimize the cost function using **gradient descent**:
θj := θj - α * ∂J(θ) / ∂θj
where:
- α → learning rate  
- ∂J(θ)/∂θj → gradient of the cost function  

For linear regression, the gradient is: ∂J(θ) / ∂θj = (1/m) Σ ( hθ(xᶦ) - yᶦ ) * xjᶦ

### 4. Normal Equation (Analytical Solution)
Instead of gradient descent, parameters can be directly computed:
θ = (XᵀX)⁻¹ Xᵀy
where:
- X → input feature matrix (with bias term)  
- y → target vector  
This method works well for small datasets but becomes computationally expensive for very large feature sets.


## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository, open issues, and submit pull requests.






