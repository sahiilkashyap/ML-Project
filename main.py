import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sns.set_theme(style='whitegrid', palette='Set2')
sns.set_context("talk")

data = pd.read_csv("winequality-red (1).csv")

print(data.head())
print(data.describe())
print(data.info())
print("Missing values?\n", data.isna().any())

plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='volatile acidity', data=data, palette='coolwarm')
plt.title("Volatile Acidity vs Wine Quality")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='citric acid', data=data, palette='viridis')
plt.title("Citric Acid vs Wine Quality")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='residual sugar', data=data, palette='magma')
plt.title("Residual Sugar vs Wine Quality")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='chlorides', data=data, palette='crest')
plt.title("Chlorides vs Wine Quality")
plt.show()

data['alcohol_bin'] = pd.cut(data['alcohol'], bins=10)
plt.figure(figsize=(12, 6))
sns.barplot(x='alcohol_bin', y='quality', data=data, palette='rocket')
plt.xticks(rotation=45)
plt.title("Alcohol (Binned) vs Wine Quality")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=data, palette='pastel')
plt.title("Count of Wine Quality Ratings")
plt.show()

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']
x = data[features]
y = data['quality']

sns.pairplot(data, x_vars=features, y_vars='quality', kind='reg', height=4, aspect=0.6, palette='husl')

plt.figure(figsize=(10, 6))
sns.barplot(x='fixed acidity', y='quality', data=data, palette='flare')
plt.title("Fixed Acidity vs Wine Quality")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

accuracy = regressor.score(x_test, y_test)
print("\nModel Accuracy: {}%".format(int(round(accuracy * 100))))
