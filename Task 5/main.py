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
CSV_FILENAME = "Automatic_Traffic_Recorder_ATR_Stations.csv"

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

# Missing values
print("\n🔍 Top 10 columns by missing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Show available columns
print("\n🧾 Available columns:")
print(list(df.columns))

# -------------------- Visualizations --------------------

# 1. Confidence level distribution
plt.figure()
sns.countplot(x='CONFIDENCE', data=df, palette='Set2', legend=False)
plt.title("Distribution of Confidence Levels")
plt.xlabel("Confidence Level")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Facility Class Distribution
plt.figure()
sns.countplot(y='FCLASS', data=df, order=df['FCLASS'].value_counts().index, palette='pastel', legend=False)
plt.title("Facility Class Distribution")
plt.xlabel("Count")
plt.ylabel("FCLASS")
plt.tight_layout()
plt.show()

# 3. Stations by State (STPOSTAL)
plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='STPOSTAL', order=df['STPOSTAL'].value_counts().index, palette='coolwarm', legend=False)
plt.title("Traffic Recorder Stations by State")
plt.xlabel("State")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# -------------------- Geospatial Hotspots --------------------
print("🗺️ Plotting traffic recorder station map...")

df_geo = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
geometry = [Point(xy) for xy in zip(df_geo['LONGITUDE'], df_geo['LATITUDE'])]
gdf = gpd.GeoDataFrame(df_geo, geometry=geometry)
gdf.set_crs(epsg=4326, inplace=True)
gdf = gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(14, 14))
gdf.plot(column='CONFIDENCE', cmap='Reds', alpha=0.6, markersize=10, legend=True, ax=ax)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_axis_off()
plt.title("Traffic Recorder Station Map (by Confidence Level)")
plt.tight_layout()
plt.show()

print("✅ Traffic data analysis and geospatial visualization complete.")
