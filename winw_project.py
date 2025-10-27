
# 🍷 WINE QUALITY PREDICTION (RED & WHITE)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# Load datasets
red_wine = pd.read_csv(r"C:\Users\Neha\Desktop\wine_quality_analysis\winequality-red.csv", sep=';')
white_wine = pd.read_csv(r"C:\Users\Neha\Desktop\wine_quality_analysis\winequality-white.csv", sep=';')

# Add a 'type' column
red_wine['type'] = 'red'
white_wine['type'] = 'white'

# Combine datasets
wine = pd.concat([red_wine, white_wine], ignore_index=True)

print("✅ Dataset loaded successfully!")
print(wine.head())

# Check for missing values
print("\n🔍 Missing values per column:")
print(wine.isnull().sum())

# Basic statistics
print("\n📊 Statistical Summary:")
print(wine.describe())

# -------------------------------------
# 1️⃣ Basic Visualizations
# -------------------------------------
# Distribution of wine quality
plt.figure(figsize=(8,5))
sns.countplot(x='quality', hue='quality', data=wine, palette='mako', legend=False)
plt.title("Distribution of Wine Quality Ratings")
plt.xlabel("Quality Score")
plt.ylabel("Number of Samples")
plt.show()


# -------------------------------------
# 2️⃣ Outlier Removal
# -------------------------------------
z_scores = np.abs(stats.zscore(wine.select_dtypes(include=np.number)))
wine = wine[(z_scores < 3).all(axis=1)]
print("\n✅ Outliers removed successfully.")
print("Remaining rows after cleaning:", len(wine))
# Pairplot for EDA (sampled)
sns.pairplot(wine.sample(300), hue='type', vars=['alcohol', 'volatile acidity', 'sulphates', 'citric acid'])
plt.suptitle("Pairplot of Key Features (Sample of 300 Wines)", y=1.02)
plt.show()

# -------------------------------------
# 3️⃣ Additional Visualizations
# -------------------------------------
# ============================================
# 📊 CORRELATION HEATMAPS (RED, WHITE, & COMBINED)
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'wine', 'red_wine', and 'white_wine' exist and contain only numeric data
# Drop the non-numeric column 'type' before correlation calculation

# --- Combined wine dataset ---
plt.figure(figsize=(12, 8))
sns.heatmap(wine.drop(columns=['type']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Overall Correlation Heatmap - Red & White Wine Combined", fontsize=14)
plt.xlabel("Wine Features")
plt.ylabel("Wine Features")
plt.tight_layout()
plt.show()

# --- Red wine dataset ---
plt.figure(figsize=(10, 6))
sns.heatmap(red_wine.corr(numeric_only=True), annot=True, cmap='Reds', fmt=".2f")
plt.title("Correlation Heatmap - Red Wine", fontsize=14)
plt.xlabel("Wine Features")
plt.ylabel("Wine Features")
plt.tight_layout()
plt.show()

# --- White wine dataset ---
plt.figure(figsize=(10, 6))
sns.heatmap(white_wine.corr(numeric_only=True), annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Heatmap - White Wine", fontsize=14)
plt.xlabel("Wine Features")
plt.ylabel("Wine Features")
plt.tight_layout()
plt.show()

# Scatter plot: Alcohol vs Quality
plt.figure(figsize=(8,5))
sns.scatterplot(x='alcohol', y='quality', hue='type', data=wine, palette='Set1', alpha=0.7)
plt.title("Alcohol vs Wine Quality")
plt.xlabel("Alcohol %")
plt.ylabel("Quality Score")
plt.show()

# Scatter plot: Volatile Acidity vs Quality
plt.figure(figsize=(8,5))
sns.scatterplot(x='volatile acidity', y='quality', hue='type', data=wine, palette='Set2', alpha=0.7)
plt.title("Volatile Acidity vs Wine Quality")
plt.xlabel("Volatile Acidity")
plt.ylabel("Quality Score")
plt.show()
# Scatter plot matrix (pairplot) of selected features
selected_features = ['fixed acidity', 'citric acid', 'residual sugar', 'alcohol', 'quality']
sns.pairplot(wine[selected_features + ['type']], hue='type', palette='coolwarm', diag_kind='kde', height=2.5)
plt.suptitle("Scatterplot Matrix of Key Features", y=1.02)
plt.show()
 # Boxplot: Alcohol in Red vs White
plt.figure(figsize=(8,5))
sns.boxplot(x='type', y='alcohol', data=wine, palette='coolwarm')
plt.title("Alcohol Content in Red vs White Wine")
plt.xlabel("Wine Type")
plt.ylabel("Alcohol Percentage")
plt.show()

#------------------------------------
# MODEL DEVELOPMENT AND EVALUATION
#------------------------------------

# Features and target
X = wine.drop(['quality', 'type'], axis=1)
y = wine['quality']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📈 MODEL PERFORMANCE:")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")
