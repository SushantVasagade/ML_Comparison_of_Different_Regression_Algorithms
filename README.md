# Comparison of Different Regression Algorithms

## 📌 Overview

This project implements and compares multiple **regression algorithms** using the California Housing dataset.

The main purpose of this experiment is to understand how different machine learning regression algorithms perform on the same dataset and to compare their performance using standard regression evaluation metrics.

The following regression algorithms are implemented:

- Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression (SVR)

The **California Housing dataset** is used for this experiment. Each algorithm is trained and evaluated using the same dataset split so that their performance can be compared systematically.

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-Fold Cross-Validation

---

## 🎯 Aim

To implement and compare different **regression algorithms** using a suitable dataset and evaluate their performance using **MAE, MSE, RMSE, and R² Score**.

---

## 🎯 Objectives

The main objectives of this experiment are:

1. To understand the concept of regression in machine learning.
2. To load and explore a suitable regression dataset.
3. To perform data preprocessing and exploratory analysis.
4. To separate input features and the target variable.
5. To divide the dataset into training and testing sets.
6. To implement different regression algorithms.
7. To train the regression models using training data.
8. To generate predictions using the trained models.
9. To evaluate each model using MAE, MSE, RMSE, and R² Score.
10. To perform cross-validation for reliable model evaluation.
11. To compare the performance of different regression algorithms.
12. To understand the strengths and limitations of different regression approaches.

---

## 🧠 Theory

### What is Regression?

**Regression** is a supervised machine learning technique used to predict a continuous numerical value.

Unlike classification, where the output is a discrete class label, regression produces a continuous numerical prediction.

Examples of regression problems include:

- House price prediction
- Temperature prediction
- Sales forecasting
- Salary prediction
- Stock price estimation
- Demand forecasting

The general objective of regression is to learn a relationship between input features and a continuous target variable.

---

## 📊 Dataset Used

The **California Housing dataset** is used in this experiment.

The dataset is loaded using Scikit-learn:

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
```

The dataset contains:

- **20,640 observations**
- **8 input features**
- **1 target variable**

### Input Features

The eight input features are:

- `MedInc` — Median income
- `HouseAge` — Median house age
- `AveRooms` — Average number of rooms
- `AveBedrms` — Average number of bedrooms
- `Population` — Block population
- `AveOccup` — Average number of household members
- `Latitude` — Latitude of the district
- `Longitude` — Longitude of the district

### Target Variable

The target variable is:

`MedHouseVal`

`MedHouseVal` represents the median house value for the corresponding California district.

---

## 🔄 Machine Learning Workflow

The complete machine learning workflow used in this experiment is:

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Algorithm Comparison
```

---

## 🧹 Data Exploration and Preprocessing

Before training the regression models, the dataset is explored to understand its structure and quality.

The following operations are performed:

- Loading the dataset
- Creating a DataFrame
- Displaying the first records
- Checking dataset shape
- Checking column information
- Generating descriptive statistics
- Checking missing values
- Checking duplicate records
- Separating features and target

These steps help understand the dataset before applying regression algorithms.

---

## 🎯 Feature and Target Separation

The dataset is divided into two parts.

### Features

The eight input variables are used as independent features:

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`

### Target

The target variable is:

`MedHouseVal`

The general structure is:

```text
X → Input Features
y → Target Variable
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing subsets.

In this experiment:

- **80% → Training Data**
- **20% → Testing Data**

The training dataset is used to train the regression models, while the testing dataset is used to evaluate their performance on unseen observations.

A fixed random state is used:

`random_state = 42`

This ensures that the same train-test split can be reproduced.

---

# 🤖 Regression Algorithms

## 1. Linear Regression

**Linear Regression** is one of the simplest and most widely used regression algorithms.

It assumes a linear relationship between the input features and the target variable.

For a single feature, the general equation is:

```text
ŷ = b₀ + b₁x
```

For multiple features:

```text
ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

where:

- `ŷ` = predicted value
- `b₀` = intercept
- `b₁, b₂, ..., bₙ` = model coefficients
- `x₁, x₂, ..., xₙ` = input features

The implementation uses:

```python
from sklearn.linear_model import LinearRegression
```

The model is created using:

```python
LinearRegression()
```

### Characteristics

- Simple and easy to understand
- Fast to train
- Easy to interpret
- Assumes a linear relationship
- Useful as a baseline regression model

---

## 2. Polynomial Regression

**Polynomial Regression** extends Linear Regression by introducing polynomial features.

It can model non-linear relationships between input features and the target variable.

In this experiment, **degree-2 polynomial features** are used.

A simple polynomial equation can be represented as:

```text
y = b₀ + b₁x + b₂x²
```

The implementation uses:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
```

The polynomial feature transformation uses:

```python
PolynomialFeatures(degree=2)
```

The transformed features are then used with Linear Regression.

### Characteristics

- Can model non-linear relationships
- Extends the idea of linear regression
- Polynomial degree controls model complexity
- Higher degrees can increase the risk of overfitting

---

## 3. Decision Tree Regression

**Decision Tree Regression** uses a tree structure to predict continuous numerical values.

The data is recursively divided according to feature-based conditions.

The implementation uses:

```python
from sklearn.tree import DecisionTreeRegressor
```

The model used in this experiment is:

```python
DecisionTreeRegressor(
    max_depth=15,
    random_state=42
)
```

### Characteristics

- Can model non-linear relationships
- Does not require feature scaling
- Can capture complex patterns
- Easy to understand for smaller trees
- Deep trees may overfit the training data

---

## 4. Random Forest Regression

**Random Forest Regression** is an ensemble learning technique that combines multiple decision trees.

Each tree produces a prediction, and the predictions are combined to produce the final output.

The implementation uses:

```python
from sklearn.ensemble import RandomForestRegressor
```

The model used in this experiment is:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Here, `n_estimators = 100` means that the forest contains 100 decision trees.

### Characteristics

- Ensemble of multiple decision trees
- Can model complex non-linear relationships
- Generally more robust than a single decision tree
- Does not require feature scaling
- Can require more computational resources

---

## 5. Support Vector Regression (SVR)

**Support Vector Regression (SVR)** is the regression version of Support Vector Machine.

SVR attempts to find a function that predicts target values while maintaining an acceptable margin of error.

In this experiment, an **RBF (Radial Basis Function) kernel** is used.

The implementation uses:

```python
from sklearn.svm import SVR
```

The model used is:

```python
SVR(
    kernel="rbf",
    C=100,
    epsilon=0.1
)
```

### Important Parameters

#### Kernel

```text
kernel = "rbf"
```

The RBF kernel allows SVR to model non-linear relationships.

#### C

```text
C = 100
```

The `C` parameter controls the trade-off between model complexity and tolerance for errors.

#### Epsilon

```text
epsilon = 0.1
```

The epsilon parameter defines a margin around the regression function within which errors are not penalized.

### Characteristics

- Can model non-linear relationships
- Uses kernel functions
- Sensitive to feature scaling
- Can be computationally expensive for large datasets
- Requires appropriate hyperparameter selection

---

# 📈 Regression Evaluation Metrics

After training the models, predictions are generated using the test dataset.

The models are compared using four important regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 1. Mean Absolute Error (MAE)

**Mean Absolute Error** measures the average absolute difference between actual and predicted values.

The formula is:

```text
MAE = (1/n) × Σ |yᵢ - ŷᵢ|
```

where:

- `yᵢ` = actual value
- `ŷᵢ` = predicted value
- `n` = number of observations

### Interpretation

A lower MAE indicates that the predictions are closer to the actual values on average.

---

## 2. Mean Squared Error (MSE)

**Mean Squared Error** calculates the average squared difference between actual and predicted values.

The formula is:

```text
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

Because the errors are squared, larger errors have a greater influence on the metric.

### Interpretation

A lower MSE indicates lower prediction error.

---

## 3. Root Mean Squared Error (RMSE)

**Root Mean Squared Error** is the square root of MSE.

The formula is:

```text
RMSE = √MSE
```

RMSE is expressed in the same units as the target variable.

### Interpretation

A lower RMSE indicates that the predictions are closer to the actual values.

---

## 4. R² Score

**R² Score**, also called the coefficient of determination, measures how much of the variation in the target variable is explained by the model.

The formula is:

```text
R² = 1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]
```

where:

- `yᵢ` = actual value
- `ŷᵢ` = predicted value
- `ȳ` = mean of actual target values

### Interpretation

- A higher R² generally indicates better explanatory performance.
- An R² value closer to `1` indicates that the model explains a larger proportion of the variation in the target.
- R² should be interpreted together with other evaluation metrics.

---

# 🔁 Cross-Validation

Cross-validation is used to evaluate the stability of a machine learning model across multiple subsets of the data.

In this experiment, **5-fold shuffled K-Fold cross-validation** is used.

The dataset is divided into five folds.

The process is repeated so that each fold is used as the validation set once.

The K-Fold configuration uses:

```text
n_splits = 5
shuffle = True
random_state = 42
```

### Advantages of Cross-Validation

- Uses multiple train-validation combinations
- Provides a more reliable estimate of model performance
- Reduces dependence on a single train-test split
- Helps identify model stability

---

# 📊 Model Comparison

The performance of all regression algorithms is compared using:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | Calculated in notebook | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| Polynomial Regression | Calculated in notebook | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| Decision Tree Regression | Calculated in notebook | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| Random Forest Regression | Calculated in notebook | Calculated in notebook | Calculated in notebook | Calculated in notebook |
| Support Vector Regression | Calculated in notebook | Calculated in notebook | Calculated in notebook | Calculated in notebook |

The exact values are generated when the Jupyter Notebook is executed.

---

## 📊 Interpretation of Metrics

For **MAE, MSE, and RMSE**:

```text
Lower value → Lower prediction error
```

For **R² Score**:

```text
Higher value → Better explanatory performance
```

Therefore, the regression algorithms should be compared using multiple appropriate metrics rather than relying on only one metric.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Jupyter Notebook | Experiment development |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning algorithms and evaluation |

---

# 📂 Project Structure

```text
ML_Comparison_of_Different_Regression_Algorithms/
│
├── Comparison_of_Different_Regression_Algorithms.ipynb
├── .gitignore
└── README.md
```

---

# 💻 Requirements

Install the required Python libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

If you are using the Anaconda environment:

```bash
conda activate ml
```

Then launch Jupyter Notebook:

```bash
jupyter notebook
```

---

# ▶️ How to Run

## Step 1: Clone the Repository

```bash
git clone https://github.com/SushantVasagade/ML_Comparison_of_Different_Regression_Algorithms.git
```

## Step 2: Open the Project

```bash
cd ML_Comparison_of_Different_Regression_Algorithms
```

## Step 3: Activate the Environment

```bash
conda activate ml
```

## Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

## Step 5: Run the Notebook

Open:

`Comparison_of_Different_Regression_Algorithms.ipynb`

and execute the cells sequentially.

---

# 📌 Key Learning Outcomes

After completing this experiment, the following concepts are understood:

- Supervised learning
- Regression
- Continuous target prediction
- Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression
- Feature and target separation
- Train-test splitting
- Model training
- Model prediction
- MAE
- MSE
- RMSE
- R² Score
- K-Fold Cross-Validation
- Model comparison

---

# 🔍 Algorithm Comparison

| Algorithm | Main Concept | Non-Linear Relationships | Scaling |
|---|---|---|---|
| Linear Regression | Linear relationship | Limited | Generally not required |
| Polynomial Regression | Polynomial relationship | Yes | Depends on implementation |
| Decision Tree Regression | Tree-based splitting | Yes | Not required |
| Random Forest Regression | Ensemble of decision trees | Yes | Not required |
| SVR | Margin-based regression with kernel | Yes | Important |

---

# ⚠️ Important Considerations

Regression model performance depends on several factors, including:

- Dataset characteristics
- Feature quality
- Data preprocessing
- Feature engineering
- Hyperparameter selection
- Training and testing split
- Cross-validation strategy
- Evaluation metrics

Different regression algorithms make different assumptions and use different approaches to learn relationships between features and the target.

Therefore, model performance should be evaluated systematically using multiple appropriate metrics.

---

# 🌍 Real-World Applications of Regression

Regression algorithms are widely used in:

- House price prediction
- Sales forecasting
- Demand prediction
- Temperature prediction
- Salary prediction
- Financial forecasting
- Energy consumption prediction
- Real estate analysis
- Business forecasting
- Risk estimation

---

# 📝 Conclusion

Different regression algorithms were implemented and compared using the **California Housing dataset**.

The experiment demonstrated the complete regression workflow, including data loading, exploration, preprocessing, feature-target separation, train-test splitting, model training, prediction, evaluation, and cross-validation.

Five regression algorithms were studied:

1. Linear Regression
2. Polynomial Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Support Vector Regression

The models were evaluated using **MAE, MSE, RMSE, and R² Score**. In addition, **5-fold shuffled K-Fold cross-validation** was used to obtain a more reliable assessment of model performance.

The experiment demonstrates that different regression algorithms can model relationships in different ways and can produce different prediction errors on the same dataset. Therefore, systematic evaluation using multiple metrics and cross-validation is important when comparing regression models.

---

# 👨‍💻 Author

**Sushant Vasagade**

B.Tech – Information Technology

Government College of Engineering, Karad

---

# 📚 Repository

[ML_Comparison_of_Different_Regression_Algorithms](https://github.com/SushantVasagade/ML_Comparison_of_Different_Regression_Algorithms)

---

# 📚 Machine Learning Lab

**Experiment:** Comparison of Different Regression Algorithms

**Domain:** Machine Learning

**Dataset:** California Housing Dataset

**Algorithms:** Linear Regression, Polynomial Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression (SVR)

**Evaluation Metrics:** MAE, MSE, RMSE, R² Score
