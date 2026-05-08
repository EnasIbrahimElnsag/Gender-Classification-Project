# 👤 Gender Classification — ML Pipeline

> A complete end-to-end machine learning project that predicts gender based on **physical facial features** using **5 classification algorithms**, with full preprocessing, cross-validation, visualization, and model comparison.

---

## 📁 Project Structure

```
gender-classification/
│
├── Gender_Classification.ipynb     # Main notebook (15 sections)
├── gender_classification.csv       # Dataset (5001 rows)
└── README.md
```

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| **Total Rows** | 5,001 |
| **After Removing Duplicates** | cleaned automatically |
| **Features** | 7 |
| **Target Column** | `gender` → Male (1) / Female (0) |

### 🔢 Features Description

| Feature | Type | Description |
|---------|------|-------------|
| `long_hair` | Binary (0/1) | Does the person have long hair? |
| `forehead_width_cm` | Continuous | Forehead width in cm |
| `forehead_height_cm` | Continuous | Forehead height in cm |
| `nose_wide` | Binary (0/1) | Is the nose wide? |
| `nose_long` | Binary (0/1) | Is the nose long? |
| `lips_thin` | Binary (0/1) | Are the lips thin? |
| `distance_nose_to_lip_long` | Binary (0/1) | Long distance from nose to lip? |

### 🎯 Target Encoding
```
Male   →  1
Female →  0
```

---

## 🔁 Project Workflow

```
📂 Load Data
    ↓
🔍 Data Understanding (shape, describe, info)
    ↓
❓ Check Missing Values
    ↓
🔁 Remove Duplicates
    ↓
🔠 Encode Target Column (Male=1 / Female=0)
    ↓
📊 Data Visualization
    ↓
✂️ Train-Test Split (80% / 20%) + stratify=y
    ↓
⚖️ StandardScaler (normalize features)
    ↓
🤖 Train 5 Models
    ↓
📈 Evaluate each model (Accuracy + Confusion Matrix + Learning Curve)
    ↓
🏆 Compare all models side by side
    ↓
📝 Conclusion
```

---

## 📊 Data Visualization (Section 7)

The notebook includes the following visualizations:

- **Correlation Heatmap** — shows relationships between all features using `coolwarm` colormap
- **Gender Class Distribution** — countplot to check class balance
- **Feature Distributions** — histogram for all 7 features
- **Pairplot** — pairwise relationships between all features colored by gender
- **Boxplots** — `forehead_width_cm` and `forehead_height_cm` vs gender

---

## ✂️ Preprocessing (Section 8)

```python
# Train-Test Split with stratification (keeps class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Cross-Validation Setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

> `stratify=y` ensures that the train and test sets have the same Male/Female ratio as the original dataset.

---

## 🤖 Models

### 🔵 1. Logistic Regression
- Simple linear classifier
- Evaluated with: Accuracy, Confusion Matrix, Classification Report, Cross-Validation
- Best for: linearly separable data, fast baseline

### 🌳 2. Decision Tree
- Uses feature thresholds to split data into branches
- Parameters: `max_depth=3`, `min_samples_leaf=3`, `random_state=42`
- Includes: Cross-Validation (K-Fold=5), Learning Curve, Tree Structure visualization
- Best for: interpretable models, non-linear boundaries

### 🟡 3. Naive Bayes (GaussianNB)
- Probabilistic classifier based on Bayes' theorem
- Assumes features are independent
- Includes: Cross-Validation, Learning Curve
- Best for: fast training, small datasets

### 🟠 4. KNN (K-Nearest Neighbors)
- Classifies based on the majority class of k nearest neighbors
- Parameters: `n_neighbors=3`
- Includes: Cross-Validation, Learning Curve, **Decision Boundary visualization using PCA (2D)**
- Best for: non-linear boundaries, small-to-medium datasets

### 🔴 5. SVM (Support Vector Machine)
- Finds the optimal hyperplane with maximum margin
- Parameters: `kernel='linear'`
- Includes: Cross-Validation, Learning Curve
- Best for: high-dimensional data, robust to outliers

---

## 📈 Evaluation Methods

Each model is evaluated using:

| Method | What it measures |
|--------|-----------------|
| **Accuracy Score** | Overall correct predictions |
| **Confusion Matrix** | True/False Positives and Negatives |
| **Classification Report** | Precision, Recall, F1-score per class |
| **Cross-Validation (K-Fold=5)** | Robust performance across different data splits |
| **Learning Curve** | Detects Overfitting / Underfitting |

---

## 🏆 Model Comparison (Section 14)

All 5 models are compared side by side in:
- A **sorted table** showing accuracy for each model
- A **bar chart** with the best model highlighted in 🥇 gold

---

## 📝 Key Takeaways (Section 15)

- All models achieved high accuracy due to clear feature separability between genders
- **StandardScaler** was essential for distance-based models (KNN, SVM)
- **Cross-Validation** provided more reliable evaluation than a single train/test split
- **Learning Curves** showed no significant overfitting across all models
- **PCA** was used to visualize the KNN decision boundary in 2D

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML models & evaluation |
| `imbalanced-learn` | Pipeline support |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/EnasIbrahimElnsag/gender-classification.git

# 2. Navigate to the folder
cd gender-classification

# 3. Install requirements
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# 4. Open the notebook
jupyter notebook Gender_Classification.ipynb
```

---

## 👩‍💻 Author

**Enas Ibrahim**
[![GitHub](https://img.shields.io/badge/GitHub-EnasIbrahimElnsag-black?logo=github)](https://github.com/EnasIbrahimElnsag)
