# 🚀 Gender Classification Using Machine Learning

> A complete Machine Learning classification project that predicts gender based on facial features using different classification algorithms and selecting the best performing model.
> Developed as part of the Machine Learning course to apply classification techniques, compare multiple models, and build an interactive prediction application.

\---

## 🎯 Objective

Classify gender based on facial features and determine the **best machine learning model** that achieves the highest prediction accuracy.

**Target Classes:**

* 👨 Male
* 👩 Female

\---

## 📂 Dataset

The dataset contains facial feature measurements used to classify samples into two gender classes:

|Feature|Description|
|-|-|
|Long Hair|Indicates whether the person has long hair|
|Forehead Width|Width measurement of the forehead|
|Forehead Height|Height measurement of the forehead|
|Nose Wide|Indicates nose width|
|Nose Long|Indicates nose length|
|Thin Lips|Indicates whether lips are thin|
|Distance Between Nose and Lip|Distance measurement between nose and lip|

\---

## 🔄 Project Workflow

### 1\. Data Understanding

* 📋 Loaded and explored the dataset
* 🔍 Checked dataset shape and feature information
* 📊 Studied feature distributions
* 🔗 Analyzed the relationship between features and target variable

### 2\. Data Preprocessing

* ✅ Checked for missing values
* ✅ Checked for duplicate records
* ✅ Prepared features and target variable
* ✅ Encoded the target labels
* ✅ Applied feature scaling using **StandardScaler**

\---

## 📊 Exploratory Data Analysis (EDA)

Performed different analysis techniques:

* 🔗 Correlation Analysis
* 📈 Feature relationship analysis
* 📉 Data distribution visualization

The analysis helped understand the importance of each feature before model training.

\---

## 🤖 Machine Learning Models

Several classification algorithms were trained and compared:

### 🔹 Logistic Regression

A linear classification model used to estimate the probability of each class.

### 🔹 Decision Tree Classifier

A tree-based model that learns decision rules from the features, optimized to reduce overfitting:

```python
DecisionTreeClassifier(
    max\_depth=3,
    min\_samples\_leaf=3
)
```

### 🔹 K-Nearest Neighbors (KNN)

A distance-based algorithm that predicts the class based on the nearest samples.

### 🔹 Support Vector Machine (SVM)

A powerful classifier that finds the best decision boundary between classes.

### 🔹 Naive Bayes

A probabilistic classifier based on Bayes' theorem. **Gaussian Naive Bayes achieved the best performance** on this dataset.

\---

## 📈 Model Evaluation

Models were evaluated using:

* **Accuracy Score**
* **Cross Validation**
* **Confusion Matrix**

### Results

|Model|Accuracy|
|-|-|
|**Naive Bayes** ✅|**95.83%**|
|KNN|94.44%|
|Logistic Regression|94.28%|
|SVM|94.13%|
|Decision Tree|93.97%|

> 🏆 \*\*Best Model: Gaussian Naive Bayes\*\* — achieved the highest accuracy of \*\*95.83%\*\*.
> The model performed well because the facial features provided strong distinguishable patterns between the two classes.

\---

## 🚀 Interactive Application

A **Gradio-powered** web application was developed to make the model accessible:

* 🖊️ Users enter facial feature values
* ⚡ Real-time gender prediction using the trained model
* 💡 User-friendly and intuitive interface
* 🌐 Deployed on Hugging Face Spaces for public access

### 🔗 Live Demo

👉 *https://huggingface.co/spaces/Enasibrahim/Gender\_Classification*

\---

## 🛠️ Technologies Used

|Library|Purpose|
|-|-|
|Python|Core programming language|
|Pandas|Data loading, cleaning, and manipulation|
|NumPy|Numerical computations|
|Scikit-learn|Model implementation and evaluation|
|Matplotlib|Data visualization|
|Seaborn|Statistical data visualization|
|Gradio|Interactive web application interface|

\---

## 📁 Project Structure

```
├── Gender\_Classification.ipynb
├── gender.csv
├── model.pkl
├── app.py
├── requirements.txt
└── README.md
```

\---

## 👥 Team Members

|#|Name|
|-|-|
|1|Enas Ibrahim Ali Elnsag|
|2|Malak Tamer Mohamed Ali|
|3|Salma Amer Ahmed Abdel Fattah|
|4|Fatma Mohamed Helmy Mohamed|
|5|Mariem Medhat Afifi|

\---

## ✅ Conclusion

This project demonstrates the complete machine learning workflow:

**Data Understanding → Preprocessing → EDA → Model Training → Evaluation → Deployment**

The final deployed model classifies gender from facial features with high accuracy, showcasing the power of classical Machine Learning techniques on structured data.

\---

<p align="center">
  Machine Learning Course Project \&nbsp;•\&nbsp; Classification Analysis \&nbsp;•\&nbsp; 2025
</p>

<p align="center">
  <code>#MachineLearning</code> \&nbsp;
  <code>#DataScience</code> \&nbsp;
  <code>#Python</code> \&nbsp;
  <code>#Classification</code> \&nbsp;
  <code>#NaiveBayes</code> \&nbsp;
  <code>#Gradio</code>
</p>

