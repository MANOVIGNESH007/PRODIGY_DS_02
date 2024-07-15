import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
titanic_df = pd.read_csv('train.csv')

# Display the first few rows
print(titanic_df.head())

# Check for missing values
print(titanic_df.isnull().sum())

# Fill missing values in 'Age' with the median age
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common port
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to too many missing values
titanic_df.drop('Cabin', axis=1, inplace=True)

# Verify there are no more missing values
print(titanic_df.isnull().sum())

# Check data types
print(titanic_df.dtypes)

# Convert 'Survived' to category type
titanic_df['Survived'] = titanic_df['Survived'].astype('category')

# Convert 'Pclass' to category type
titanic_df['Pclass'] = titanic_df['Pclass'].astype('category')

# Verify changes
print(titanic_df.dtypes)

# Display summary statistics
print(titanic_df.describe(include='all'))

# Visualizing the Data
sns.set_style('whitegrid')

# Countplot of survivors
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df, x='Survived')
plt.title('Count of Survivors')
plt.savefig('count_of_survivors.png')
plt.show()

# Countplot of passengers by class
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df, x='Pclass')
plt.title('Count of Passengers by Class')
plt.savefig('count_of_passengers_by_class.png')
plt.show()

# Age distribution
plt.figure(figsize=(10, 8))
sns.histplot(titanic_df['Age'], kde=True)
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.show()

# Boxplot of age by survival status
plt.figure(figsize=(10, 8))
sns.boxplot(data=titanic_df, x='Survived', y='Age')
plt.title('Age Distribution by Survival Status')
plt.savefig('age_distribution_by_survival_status.png')
plt.show()

# Barplot of survival by class
plt.figure(figsize=(10, 8))
sns.barplot(data=titanic_df, x='Pclass', y='Survived')
plt.title('Survival Rate by Class')
plt.savefig('survival_rate_by_class.png')
plt.show()

# Pairplot to visualize relationships between multiple variables
sns.pairplot(titanic_df[['Survived', 'Age', 'SibSp', 'Parch', 'Fare']], hue='Survived', diag_kind='kde')
plt.savefig('pairplot.png')
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(12, 10))
# Select only numerical columns
numerical_df = titanic_df.select_dtypes(include=[np.number])
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

# Save the cleaned dataset
titanic_df.to_csv('titanic_cleaned.csv', index=False)
