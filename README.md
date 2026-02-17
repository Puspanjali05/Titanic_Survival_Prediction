# 🚢 Titanic Survival Prediction using Machine Learning

## 📌 Project Overview

This project builds a **Supervised Machine Learning classification model** to predict whether a passenger survived the Titanic disaster.

Using real-world passenger data, the project demonstrates:

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Training  
- Model Evaluation  
- Performance Analysis  

The final model uses **Logistic Regression**, a widely used baseline algorithm for binary classification problems.

---

## 🎯 Problem Statement

Given passenger attributes such as age, gender, class, and fare, predict whether the passenger survived the Titanic disaster.

**Target Variable**
- `Survived`
  - 0 → Did Not Survive  
  - 1 → Survived  

---

## 📂 Dataset Description

The dataset contains passenger-level information:

| Feature | Description |
|----------|-------------|
| PassengerId | Unique passenger ID |
| Pclass | Ticket class (1st, 2nd, 3rd) |
| Name | Passenger name |
| Sex | Gender |
| Age | Age in years |
| SibSp | No. of siblings/spouses aboard |
| Parch | No. of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation |
| Survived | Survival status (Target Variable) |

---

## 🛠️ Tech Stack

**Language**
- Python 3.x  

**Libraries Used**
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📊 Project Workflow

### 1️⃣ Data Preprocessing

- Dropped `Cabin` column (excessive missing values)  
- Filled missing `Age` values with mean  
- Filled missing `Embarked` values with mode  
- Removed irrelevant features: `PassengerId`, `Name`, `Ticket`  

---

### 2️⃣ Exploratory Data Analysis (EDA)

**Key Insights:**

- Female passengers had significantly higher survival rates.  
- First-class passengers had better survival probability.  
- Third-class passengers had the lowest survival rate.  
- Fare and passenger class influenced survival chances.  

Visualizations performed:
- Survival distribution  
- Gender vs Survival  
- Passenger Class vs Survival  
- Feature correlations  

---

### 3️⃣ Feature Engineering & Encoding

Converted categorical features into numeric format:

- `Sex`  
  - male → 0  
  - female → 1  

- `Embarked`  
  - S → 0  
  - C → 1  
  - Q → 2  

---

### 4️⃣ Train-Test Split

- 80% Training Data  
- 20% Testing Data  

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)
```

---

### 5️⃣ Model Building

Logistic Regression was used as the classification model:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

### 6️⃣ Model Evaluation

Performance metrics used:

- Accuracy Score (Training)  
- Accuracy Score (Testing)  

```python
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))
```

---

## 📈 Model Performance

- The model demonstrates good generalization.  
- Training and testing accuracy are closely aligned.  
- Gender and Passenger Class are dominant predictive features.  

Additional evaluation methods that can be added:
- Confusion Matrix  
- Precision & Recall  
- F1-Score  
- ROC-AUC Curve  

---

## 🚀 Future Improvements

- Implement Random Forest Classifier  
- Apply Gradient Boosting (XGBoost)  
- Perform Hyperparameter Tuning  
- Add Cross-Validation  
- Deploy using Flask or Streamlit  
- Create an interactive dashboard  

---

## 📁 Project Structure

```
Titanic-Survival-Prediction/
│
├── Titanic_Survival_Prediction.ipynb
├── Titanic-Dataset.csv
└── README.md
```

---

## 🧠 Key Learning Outcomes

- Hands-on experience with real-world dataset  
- Data cleaning & missing value handling  
- Exploratory Data Analysis (EDA)  
- Feature encoding techniques  
- Supervised Machine Learning (Classification)  
- Model evaluation using Scikit-learn  

---

## 👩‍💻 Author

**Puspanjali Behera**  
B.Tech – Information Technology  
Odisha University of Technology and Research, Bhubaneswar

---

⭐ If you found this project useful, consider giving it a star!
