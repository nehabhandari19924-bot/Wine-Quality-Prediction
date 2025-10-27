# Wine-Quality-Prediction

## 📖 Overview

This project focuses on predicting the quality of red and white wines using machine learning. The dataset includes various physicochemical features (like acidity, alcohol, sugar content, etc.) that influence wine quality. The project performs data cleaning, visualization, correlation analysis, and model training using Linear Regression to evaluate prediction accuracy.

## 🎯 Objectives

Analyze and visualize wine characteristics across red and white varieties.

Identify correlations between physicochemical features and wine quality.

Remove outliers to improve model performance.

Build and evaluate a machine learning model to predict wine quality scores.

Compare quality attributes between red and white wines.

## 🧩 Dataset Information

Source: UCI Machine Learning Repository - Wine Quality Dataset

Files Used:

winequality-red.csv

winequality-white.csv

## 🔬 Features 

1. Fixed Acidity – Amount of non-volatile acids (mainly tartaric acid) in wine.

2. Volatile Acidity – Amount of acetic acid; too high gives a vinegar taste.

3. Citric Acid – Adds freshness and flavor to wine.

4. Residual Sugar – Sugar remaining after fermentation.

5. Chlorides – Amount of salt in wine.

6. Free Sulfur Dioxide – Free form of SO₂ that prevents oxidation.

7. Total Sulfur Dioxide – Sum of free and bound SO₂ in the wine.

8. Density – Mass of wine per unit volume (related to sugar and alcohol content).

9. pH – Acidity level of wine (lower pH = more acidic).

10. Sulphates – Contribute to wine’s stability and flavor.

11. Alcohol – Percentage of alcohol in the wine.

12. Type – Category of wine: Red or White (added column).

## ⚙️ Technologies Used

Programming Language: Python

Libraries:

pandas, numpy – Data manipulation

matplotlib, seaborn – Data visualization

scipy – Statistical computations

sklearn – Machine learning (Linear Regression, scaling, metrics)

## 📊 Steps & Implementation
### 1️⃣ Data Loading & Preprocessing

Loaded red and white wine datasets.

Added a “type” column to distinguish between red and white wines.

Combined datasets into a single DataFrame.

Checked for missing values and summarized statistics.

### 2️⃣ Data Cleaning

Removed outliers using Z-score method.

Ensured clean and balanced data distribution for modeling.

### 3️⃣ Data Visualization

Countplot for wine quality distribution.

Pairplots for feature comparison.

Boxplots comparing alcohol levels in red vs white wine.

Scatterplots:

Alcohol vs Quality

Volatile Acidity vs Quality

Correlation Heatmaps:

For combined, red, and white datasets to study feature relationships.

### 4️⃣ Model Development

Split data into training (80%) and testing (20%) sets.

Scaled features using StandardScaler.

Implemented Linear Regression model for quality prediction.

### 5️⃣ Model Evaluation

Used standard regression metrics:

Metric	Description	Value (example)
MAE	Mean Absolute Error	~0.45
MSE	Mean Squared Error	~0.32
RMSE	Root Mean Squared Error	~0.56
R²	Coefficient of Determination	~0.67

(Values depend on your dataset after cleaning.)

## 🔍 Insights & Observations

Alcohol and citric acid have a positive correlation with wine quality.

Volatile acidity negatively impacts wine quality.

White wines generally show higher quality consistency compared to red wines.

Removing outliers improved regression accuracy.

## 🧠 Conclusion

The Linear Regression model provided a reasonable estimation of wine quality.
However, model performance could be improved by:

Using advanced models like Random Forest, XGBoost, or Neural Networks.

Performing feature selection or dimensionality reduction (e.g., PCA).

Adding more qualitative factors such as grape variety or fermentation details.

## 🚀 Future Scope

Deploy the model using Streamlit or Flask for real-time predictions.

Create an interactive dashboard for visualization.

Implement classification models to categorize wines (e.g., low, medium, high quality).

## 📂 Project Structure
wine_quality_analysis/
│
├── winequality-red.csv
├── winequality-white.csv
├── wine_quality_prediction.py
├── README.md
└── outputs/
    ├── correlation_heatmap.png
    ├── alcohol_vs_quality.png
    ├── boxplot_alcohol.png

