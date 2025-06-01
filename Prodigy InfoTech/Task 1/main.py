import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set the style of seaborn
sns.set(style="whitegrid")

# Sample data for demonstration
# Categorical data: Gender distribution
gender_data = ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'] * 10
gender_df = pd.DataFrame(gender_data, columns=['Gender'])

# Continuous data: Age distribution
np.random.seed(42)  # For reproducibility
age_data = np.random.randint(18, 70, size=1000)  # Random ages between 18 and 70
age_df = pd.DataFrame(age_data, columns=['Age'])

# Data Preprocessing
# For gender, convert to a categorical type for better performance
gender_df['Gender'] = gender_df['Gender'].astype('category')

# Create a bar chart for gender distribution using Seaborn
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
sns.countplot(data=gender_df, x='Gender', hue=None)  # Removed palette
plt.title('Gender Distribution', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Create a histogram for age distribution using Seaborn
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
sns.histplot(age_df['Age'], bins=30, kde=True, color='skyblue', stat='density')
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plots
plt.tight_layout()
plt.show()

# Interactive Plot using Plotly for Age Distribution
fig = px.histogram(age_df, x='Age', nbins=30, title='Interactive Age Distribution',
                   labels={'Age': 'Age'},
                   histnorm='probability density',
                   template='plotly_white')
fig.update_traces(marker_color='skyblue')
fig.show()

# Interactive Bar Chart using Plotly for Gender Distribution
gender_counts = gender_df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

fig_gender = px.bar(gender_counts, x='Gender', y='Count', title='Interactive Gender Distribution',
                    labels={'Count': 'Count'},
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Plasma)
fig_gender.show()
