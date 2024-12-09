
#import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Display the first few rows
print("Dataset Preview:")
print(df.head())

# Check data types and missing values
print("\nData Info (Data Types and Missing Values):")
df.info()


# Get basic statistics of numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Group by 'species' and calculate the mean of numerical columns
print("\nMean of Numerical Columns by Species:")
print(df.groupby('species').mean())

# Data Visualization

# Set seaborn style
sns.set(style="whitegrid")

# 1. Line chart (

df_sales = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
    'sales': [220, 240, 250, 210, 260, 270, 280]
})

plt.figure(figsize=(10, 6))
plt.plot(df_sales['month'], df_sales['sales'], marker='o', color='b', label='Sales')
plt.title('Sales Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

# 2. Bar chart 
plt.figure(figsize=(10, 6))
sns.barplot(x=df['species'], y=df['petal length (cm)'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram 
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True, color='green')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot 
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['sepal length (cm)'], y=df['petal length (cm)'], hue=df['species'], palette='Set2')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Error Handling 
try:
    df_invalid = pd.read_csv('invalid_dataset.csv')  
except FileNotFoundError:
    print("File not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("No data found in the file.")
except Exception as e:
    print(f"An error occurred: {e}")
