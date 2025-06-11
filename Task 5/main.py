import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

# Settings
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# Dataset path
CSV_FILENAME = "US_Accidents_March23.csv"

# Check if the dataset exists
if not os.path.exists(CSV_FILENAME):
    print(f"\n❌ Dataset CSV ({CSV_FILENAME}) not found.")
    print("Please ensure the file is downloaded from Kaggle and placed in the script directory:")
    print(f"📂 {os.path.abspath('.')}")
    raise FileNotFoundError(f"Dataset file '{CSV_FILENAME}' not found.")

# Load dataset
print("📊 Loading data...")
df = pd.read_csv(CSV_FILENAME)
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Dataset info
print("\nℹ️ Dataset Info:")
print(df.info())
print("\n🔍 Top 10 columns by missing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Define and filter only available columns
columns_of_interest = [
    'ID', 'Severity', 'Start_Time', 'End_Time',
    'Start_Lat', 'Start_Lng', 'City', 'State',
    'Weather_Condition', 'Wind_Direction', 'Temperature(F)',
    'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
    'Precipitation(in)', 'Wind_Speed(mph)',
    'Sunrise_Sunset'
]
available_columns = [col for col in columns_of_interest if col in df.columns]
df = df[available_columns]

# Convert to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df.dropna(subset=['Start_Time', 'End_Time'], inplace=True)

# Extract time features
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.day_name()
df['Month'] = df['Start_Time'].dt.month_name()

# -------------------- Visualizations --------------------

# 1. Severity Distribution
plt.figure()
sns.countplot(x='Severity', data=df, palette='Reds')
plt.title("Accident Severity Distribution")
plt.xlabel("Severity (1=Lowest, 4=Highest)")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# 2. Accidents by Hour
plt.figure()
sns.countplot(x='Hour', data=df, palette='viridis')
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# 3. Accidents by Day of Week
plt.figure()
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(x='DayOfWeek', data=df, order=order, palette='Blues')
plt.title("Accidents by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# 4. Top Weather Conditions
plt.figure(figsize=(14, 7))
top_weather = df['Weather_Condition'].value_counts().nlargest(15).index
sns.countplot(data=df[df['Weather_Condition'].isin(top_weather)],
              y='Weather_Condition', order=top_weather, palette='coolwarm')
plt.title("Top 15 Weather Conditions During Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Weather Condition")
plt.tight_layout()
plt.show()

# 5. Day vs Night Accidents Severity
plt.figure()
sns.countplot(x='Sunrise_Sunset', hue='Severity', data=df, palette='Set1')
plt.title("Day vs Night Accidents by Severity")
plt.xlabel("Sunrise or Sunset")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# -------------------- Geospatial Hotspots --------------------
print("🗺️ Plotting accident hotspots...")

df_geo = df.dropna(subset=['Start_Lat', 'Start_Lng'])
geometry = [Point(xy) for xy in zip(df_geo['Start_Lng'], df_geo['Start_Lat'])]
gdf = gpd.GeoDataFrame(df_geo, geometry=geometry)
gdf.set_crs(epsg=4326, inplace=True)
gdf = gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(14, 14))
gdf.plot(column='Severity', cmap='Reds', alpha=0.5, markersize=10, legend=True, ax=ax)
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
ax.set_axis_off()
plt.title("Accident Hotspots in US by Severity")
plt.tight_layout()
plt.show()

# -------------------- Correlation Heatmap --------------------
numeric_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                'Visibility(mi)', 'Precipitation(in)', 'Wind_Speed(mph)']
corr_data = df[numeric_cols].dropna()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_data.corr(), annot=True, cmap='RdPu', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Weather and Accident Features")
plt.tight_layout()
plt.show()

print("✅ Traffic accident data analysis and visualization complete.")
