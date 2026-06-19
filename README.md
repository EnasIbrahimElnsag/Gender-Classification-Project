# 👤 Gender Classification — ML Pipeline

A complete machine learning project that predicts gender based on physical facial features using **five classification algorithms**: Logistic Regression, Decision Tree, Naive Bayes, KNN, and SVM — with full evaluation, cross-validation, and model comparison.

\---

## 📁 Project Structure

```
├── Gender\\\\\\\_classification.ipynb   # Main Jupyter Notebook
├── gender\\\\\\\_classification.csv     # Dataset
└── README.md                     # Project documentation
```

\---

## 🎯 Objective

Predict a person's **gender (Male/Female)** based on physical facial measurements, and compare the performance of five different classification algorithms to find the best model.

\---

## 📦 Dataset

The dataset (`gender\\\\\\\_classification.csv`) contains physical facial measurements used to classify gender.

|Column|Description|
|-|-|
|`forehead\\\\\\\_width\\\\\\\_cm`|Width of the forehead in cm|
|`forehead\\\\\\\_height\\\\\\\_cm`|Height of the forehead in cm|
|`nose\\\\\\\_wide`|Whether the nose is wide (binary)|
|`nose\\\\\\\_long`|Whether the nose is long (binary)|
|`lips\\\\\\\_thin`|Whether the lips are thin (binary)|
|`distance\\\\\\\_nose\\\\\\\_to\\\\\\\_lip\\\\\\\_long`|Distance from nose to lip (binary)|
|`gender`|Target variable — Male (1) / Female (0)|

\---

## 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model\\\\\\\_selection import train\\\\\\\_test\\\\\\\_split, StratifiedKFold, cross\\\\\\\_val\\\\\\\_score, learning\\\\\\\_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear\\\\\\\_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive\\\\\\\_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy\\\\\\\_score, confusion\\\\\\\_matrix, classification\\\\\\\_report
from imblearn.pipeline import Pipeline
```

\---

## 🔄 Workflow

### 1\. Data Loading \& Understanding

* Loaded the CSV dataset using `pandas`
* Explored shape, data types, and statistical summaries

### 2\. Data Preprocessing

* ✅ Checked for **missing values** → None found
* ✅ Removed **duplicate records** using `drop\\\\\\\_duplicates()`
* 🔤 **Encoded target variable** `gender`: Male → 1, Female → 0

### 3\. Exploratory Data Analysis (EDA)

* **Correlation Heatmap** — examined relationships between all features
* **Class Distribution** — checked balance between Male/Female classes
* **Feature Distributions** — histograms for all numeric features
* **Pairplot** — explored pairwise relationships
* **Boxplots** — compared forehead width \& height across genders

### 4\. Feature Scaling \& Splitting

* **80/20 train-test split** with `stratify=y` to maintain class balance
* **StandardScaler** applied to normalize features (critical for KNN \& SVM)
* **StratifiedKFold (5 splits)** used for cross-validation

\---

## 🤖 Models

### 🔵 Model 1 — Logistic Regression

A simple linear classifier that models the probability of belonging to each class.

**Best for:** linearly separable data, fast baseline model.

```python
lr\\\\\\\_model = LogisticRegression()
```

\---

### 🌳 Model 2 — Decision Tree

Splits the data based on feature thresholds to make predictions. Used with `max\\\\\\\_depth=3` and `min\\\\\\\_samples\\\\\\\_leaf=3` to avoid overfitting.

**Best for:** interpretable models, non-linear boundaries.

```python
DecisionTreeClassifier(max\\\\\\\_depth=3, min\\\\\\\_samples\\\\\\\_leaf=3, random\\\\\\\_state=42)
```

\---

### 🟡 Model 3 — Naive Bayes

A probabilistic classifier based on Bayes' theorem that assumes feature independence.

**Best for:** fast training, works well with small datasets.

```python
nb\\\\\\\_model = GaussianNB()
```

\---

### 🟠 Model 4 — K-Nearest Neighbors (KNN)

Classifies a point based on the majority class among its K=3 nearest neighbors. Decision boundary visualized using PCA (2D).

**Best for:** simple non-linear boundaries, small-to-medium datasets.

```python
KNeighborsClassifier(n\\\\\\\_neighbors=3)
```

\---

### 🔴 Model 5 — Support Vector Machine (SVM)

Finds the optimal hyperplane that separates classes with maximum margin using a linear kernel.

**Best for:** high-dimensional data, robust to outliers.

```python
SVC(kernel='linear')
```

\---

## 📊 Model Comparison (Results)

|Rank|Model|Test Accuracy|CV Mean Accuracy|
|-|-|-|-|
|🥇 1|**Naive Bayes**|**95.83%**|**95.79%**|
|🥈 2|KNN|94.44%|94.63%|
|🥉 3|Logistic Regression|94.28%|95.17%|
|4|SVM|94.13%|94.93%|
|5|Decision Tree|93.97%|93.97%|

> 📌 \\\\\\\*\\\\\\\*Best Model:\\\\\\\*\\\\\\\* Naive Bayes achieved the highest test accuracy of \\\\\\\*\\\\\\\*95.83%\\\\\\\*\\\\\\\* on this dataset.

\---

## 📈 Evaluation Metrics

Each model was evaluated using:

|Metric|Description|
|-|-|
|**Accuracy**|Overall correct predictions / total predictions|
|**Confusion Matrix**|Visual breakdown of True/False Positives \& Negatives|
|**Classification Report**|Precision, Recall, F1-Score per class|
|**Cross-Validation**|5-Fold StratifiedKFold for robust evaluation|
|**Learning Curve**|Training vs Validation accuracy across dataset sizes|

\---

## 📉 Visualizations

Each model includes:

* **Confusion Matrix Heatmap** — shows prediction errors per class
* **Learning Curve** — detects overfitting or underfitting
* **KNN Decision Boundary** — 2D visualization using PCA

\---

## 🔍 Key Findings

1. All five models performed **very well** (above 96%) due to clear feature separability between genders.
2. **StandardScaler** was essential for distance-based models (KNN, SVM) — without it, results would be worse.
3. **Cross-validation** provided a more reliable estimate than a single train-test split.
4. **Learning Curves** showed no significant overfitting across any of the models.
5. **Forehead width** and **forehead height** were among the most distinguishing features between genders.
6. **Naive Bayes** was the best performing model with **95.83% accuracy**, despite its simplicity.

\---
---



## 🌐 Deployment \& User Interface



After selecting the best-performing model (**Naive Bayes**), we moved from the experimentation phase to building a simple interactive application.



We developed a user-friendly interface using **Gradio** that allows users to enter facial characteristics and get the predicted gender directly.



The interface takes the following inputs:



- Long Hair

- Forehead Width

- Forehead Height

- Nose Wide

- Nose Long

- Thin Lips

- Distance Between Nose and Lip



The application uses the trained **Naive Bayes model** to make predictions and display the final result (**Male/Female**).



This step helped us transform the project from a Jupyter Notebook experiment into a practical machine learning application.



🔗 Live Demo:

https://ee3dd8315d81111685.gradio.live



\---



## ✅ Conclusion

|Question|Answer|
|-|-|
|Best model?|**Naive Bayes (95.83%)**|
|Most important features?|**Forehead width \& height**|
|Any overfitting?|No — learning curves were stable|
|Was scaling necessary?|Yes — especially for KNN and SVM|
|Class balance?|Balanced — stratified split was used|

\---

## 🚀 How to Run

1. Clone or download the project files
2. Install required libraries:

```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

3. Place `gender\\\\\\\_classification.csv` in the same directory as the notebook
4. Open and run `Gender\\\\\\\_classification.ipynb` cell by cell in Jupyter Notebook or JupyterLab

\---

## \## Developed by

##### \* Enas Ibrahim Ali Elnsag

##### \* Malak Tamer Mohamed Ali

##### \* Salma Amer Ahmed Abdel Fattah

##### \* Fatma Mohamed Helmy Mohamed

##### \* Mariem Medhet Afifi

