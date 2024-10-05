import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

with zipfile.ZipFile('C:/Users/User/Downloads/titanic.zip', 'r') as z:
    print(z.namelist())

    with z.open('train.csv') as f:
        titanic_data = pd.read_csv(f)

print("Initial Data:")
print(titanic_data.head())

print("\nData Info:")
print(titanic_data.info())

print("\nMissing Values:")
print(titanic_data.isnull().sum())

titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

titanic_data.drop(columns=['Cabin'], inplace=True, errors='ignore')

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

titanic_data['Sex'] = titanic_data['Sex'].astype('category')
titanic_data['Embarked'] = titanic_data['Embarked'].astype('category')

titanic_data.drop_duplicates(inplace=True)

print("\nMissing Values After Cleaning:")
print(titanic_data.isnull().sum())

print("\nDescriptive Statistics:")
print(titanic_data.describe())

sns.set(style="whitegrid")

plt.figure(figsize=(8, 4))
sns.barplot(x='Sex', y='Survived', data=titanic_data)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(x='Pclass', y='Survived', data=titanic_data)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(titanic_data['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data=titanic_data, x='Age', hue='Survived', bins=30, kde=True, multiple='stack')
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 4))
sns.barplot(x='Embarked', y='Survived', data=titanic_data)
plt.title('Survival Rate by Embarkation Point')
plt.ylabel('Survival Rate')
plt.show()

numeric_data = titanic_data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print("\nFinal Cleaned Data:")
print(titanic_data.head())
