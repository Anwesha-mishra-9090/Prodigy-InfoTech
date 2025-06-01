import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

# Attempt to load Titanic dataset from local file
DATA_PATH = "titanic.csv"

if not os.path.exists(DATA_PATH):
    # Download Titanic data from an online source if local file not found
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print(f"Downloading Titanic dataset from {url}...")
    titanic_df = pd.read_csv(url)
    titanic_df.to_csv(DATA_PATH, index=False)
    print("Download complete.")
else:
    titanic_df = pd.read_csv(DATA_PATH)

# Preview data
print("Initial data preview:")
print(titanic_df.head())

print("\nData info and missing values:")
print(titanic_df.info())
print(titanic_df.isnull().sum())

# Data Cleaning

# Fill missing Age values using median grouped by Pclass and Sex
titanic_df['Age'] = titanic_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Fill missing Embarked values with the mode
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])

# Cabin has many missing values, create a feature "Has_Cabin" to denote presence or absence
titanic_df['Has_Cabin'] = titanic_df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)

# Dropping Cabin column (too many missing and complex)
titanic_df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

print("\nAfter cleaning, missing values:")
print(titanic_df.isnull().sum())

# Exploratory Data Analysis (EDA)

# Basic statistics
print("\nBasic Statistics:")
print(titanic_df.describe(include='all'))

# Visualizations
plt.figure(figsize=(16,12))

# Survival counts
plt.subplot(3, 2, 1)
sns.countplot(data=titanic_df, x='Survived', hue='Sex')
plt.title('Survival Counts by Gender')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Count')

# Passenger Class distribution
plt.subplot(3, 2, 2)
sns.countplot(data=titanic_df, x='Pclass')  # Removed palette
plt.title('Passenger Class Distribution')
plt.xlabel('Passenger Class')
plt.ylabel('Count')

# Age distribution by survival status
plt.subplot(3, 2, 3)
sns.histplot(data=titanic_df, x='Age', hue='Survived', multiple='stack', bins=30, palette='coolwarm')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')

# Fare distribution by survival status
plt.subplot(3, 2, 4)
sns.boxplot(data=titanic_df, x='Survived', y='Fare')
plt.title('Fare Distribution by Survival')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Fare')

# Embarked distribution by survival status
plt.subplot(3, 2, 5)
sns.countplot(data=titanic_df, x='Embarked', hue='Survived', palette='Set2')
plt.title('Embarked Port by Survival')
plt.xlabel('Embarked')
plt.ylabel('Count')

# Has Cabin (binary feature) by survival
plt.subplot(3, 2, 6)
sns.countplot(data=titanic_df, x='Has_Cabin', hue='Survived', palette='Set1')
plt.title('Cabin Presence by Survival')
plt.xlabel('Has Cabin (0=No, 1=Yes)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
# Select only numeric columns for correlation
numeric_cols = titanic_df.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Interactive EDA with Plotly

# Survival by Pclass and Sex (stacked bar)
survival_pclass_sex = titanic_df.groupby(['Pclass', 'Sex', 'Survived']).size().reset_index(name='Count')
fig1 = px.bar(survival_pclass_sex, x='Pclass', y='Count', color='Survived',
              facet_col='Sex', barmode='stack',
              title='Survival Counts by Passenger Class and Gender',
              labels={'Survived': 'Survived (0=No, 1=Yes)'})
fig1.update_layout(template='plotly_white')
fig1.show()

# Age distribution boxplot by survival using Plotly
fig2 = px.box(titanic_df, x='Survived', y='Age', color='Survived',
              title='Age Distribution by Survival (Interactive)',
              labels={'Survived':'Survived (0=No,1=Yes)', 'Age':'Age'})
fig2.update_layout(template='plotly_white')
fig2.show()

# Fare distribution violin plot by survival using Plotly
fig3 = px.violin(titanic_df, x='Survived', y='Fare', color='Survived', box=True, points='all',
                 title='Fare Distribution by Survival (Interactive)',
                 labels={'Survived':'Survived (0=No,1=Yes)', 'Fare':'Fare'})
fig3.update_layout(template='plotly_white')
fig3.show()

print("EDA Completed.")
