import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Enable inline plotting for visualizations
plt.rcParams["figure.figsize"] = (8, 5)
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Simulate missing values for demonstration
    df.loc[5:7, 'sepal length (cm)'] = np.nan

    # Handle missing values by filling with column mean
    df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean(), inplace=True)

    # Task 2: Basic Data Analysis
    basic_stats = df.describe()
    mean_sepal_by_species = df.groupby('species')['sepal length (cm)'].mean()

    # Task 3: Data Visualization
    # 1. Line Chart: Sepal Length trend by index
    plt.figure()
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.plot(subset.index, subset['sepal length (cm)'], marker='o', label=species)
    plt.title("Sepal Length Trend by Species")
    plt.xlabel("Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/data/line_chart_sepal_length.png")

    # 2. Bar Chart: Average Petal Length per Species
    plt.figure()
    sns.barplot(x='species', y='petal length (cm)', data=df)
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.tight_layout()
    plt.savefig("/mnt/data/bar_chart_petal_length.png")

    # 3. Histogram: Sepal Width Distribution
    plt.figure()
    sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
    plt.title("Distribution of Sepal Width")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("/mnt/data/histogram_sepal_width.png")

    # 4. Scatter Plot: Sepal Length vs Petal Length
    plt.figure()
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, s=80)
    plt.title("Sepal Length vs Petal Length by Species")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title='Species')
    plt.tight_layout()
    plt.savefig("/mnt/data/scatter_plot_sepal_petal.png")

    output_summary = {
        "head": df.head(),
        "data_types": df.dtypes,
        "missing_values": df.isnull().sum(),
        "basic_statistics": basic_stats,
        "mean_sepal_length_by_species": mean_sepal_by_species
    }

except Exception as e:
    output_summary = {"error": str(e)}

output_summary.keys()
