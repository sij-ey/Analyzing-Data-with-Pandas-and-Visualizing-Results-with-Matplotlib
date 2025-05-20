import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    # Convert to pandas DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Add target (species) column with names
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nChecking for missing values:")
    print(df.isnull().sum())

    # For demonstration, let's artificially create some missing values
    df.loc[5:7, 'sepal length (cm)'] = np.nan

    print("\nMissing values after adding NaNs:")
    print(df.isnull().sum())

    # Clean dataset by filling missing values with column mean
    df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean(), inplace=True)
    print("\nMissing values after filling NaNs:")
    print(df.isnull().sum())

except Exception as e:
    print(f"Error loading or cleaning dataset: {e}")

# Task 2: Basic Data Analysis
try:
    print("\nBasic statistics of numerical columns:")
    print(df.describe())

    print("\nMean sepal length by species:")
    mean_sepal_length = df.groupby('species')['sepal length (cm)'].mean()
    print(mean_sepal_length)

    # Interesting finding example:
    print("\nObservation: Setosa species tends to have smaller sepal length on average.")

except Exception as e:
    print(f"Error during data analysis: {e}")

# Task 3: Data Visualization

try:
    sns.set(style="whitegrid")

    # 1. Line chart (simulate a time series with index as time)
    plt.figure(figsize=(8, 5))
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.plot(subset.index, subset['sepal length (cm)'], marker='o', label=species)
    plt.title("Sepal Length Trend by Species (index as proxy for time)")
    plt.xlabel("Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.show()

    # 2. Bar chart: average petal length per species
    plt.figure(figsize=(6, 4))
    sns.barplot(x='species', y='petal length (cm)', data=df)
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Average Petal Length (cm)")
    plt.show()

    # 3. Histogram of sepal width to understand distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
    plt.title("Distribution of Sepal Width")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.show()

    # 4. Scatter plot: sepal length vs petal length, colored by species
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, s=70)
    plt.title("Sepal Length vs Petal Length by Species")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title='Species')
    plt.show()

except Exception as e:
    print(f"Error during plotting: {e}")
