import geopandas as gpd
import pandas as pd
import requests
import io
import zipfile
from pathlib import Path

from us_congressional_districts.utils import get_data_directory


# 1. Define the URL for the Congressional District shapefile ZIP
zip_file_url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_cd118_5m.zip"

# Column names we expect in the shapefile (based on typical Census CD files)
# GEOID column usually contains STATEFP + CD118FP (e.g., "0101" for AL Dist 1)
geoid_col_shp = 'GEOID'
name_col_shp = 'NAMELSAD' # (e.g., "Congressional District 1")

try:
    response = requests.get(zip_file_url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
    
    zip_file_bytes = io.BytesIO(response.content)
    gdf_cd = gpd.read_file(zip_file_bytes)
    
except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")
    exit()
except Exception as e:
    print(f"Error reading shapefile from downloaded ZIP content: {e}")
    print("This might happen if the ZIP structure is unexpected or the file is corrupted.")
    print("Consider downloading the file manually, unzipping it, and then loading the .shp file directly.")
    exit()

print(f"\nColumns in the GeoDataFrame: {gdf_cd.columns.tolist()}")
print(f"CRS of the GeoDataFrame: {gdf_cd.crs}")
print(gdf_cd.head())

if geoid_col_shp not in gdf_cd.columns:
    print(f"Error: Expected GEOID column '{geoid_col_shp}' not found in the shapefile.")
    print(f"Please inspect the columns above and adjust 'geoid_col_shp'.")
    exit()
if name_col_shp not in gdf_cd.columns:
    print(f"Warning: Expected NAME column '{name_col_shp}' not found. Will proceed without names if it's missing.")

# 5. Calculate Centroids
# The .centroid attribute gives the geometric center.
# For geographic data (lat/lon), these will be lat/lon coordinates.
# If you need centroids for a projected map display, you might reproject first.
# For now, we'll get the geographic centroids.
gdf_cd['geometry_centroid'] = gdf_cd.geometry.centroid
gdf_cd['x'] = gdf_cd['geometry_centroid'].x  # Typically Longitude
gdf_cd['y'] = gdf_cd['geometry_centroid'].y  # Typically Latitude

# 6. Create a DataFrame with the desired information
output_columns = ['AFFGEOID', 'x', 'y']
df_district_coords = gdf_cd[output_columns].copy()
df_district_coords.rename(
    columns={'AFFGEOID': 'GEO_ID'},
    inplace=True
)
df_district_coords.sort_values("GEO_ID", inplace=True)

# Save to the inputs directory
csv_output_path = (
    get_data_directory() / "input" / "geographies" / "districts.csv"
)
df_district_coords.to_csv(csv_output_path, index=False)


# --- Optional: Basic Plot of Districts and their Centroids ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
gdf_cd.plot(ax=ax, facecolor='lightblue', edgecolor='grey', linewidth=0.5, alpha=0.7)
ax.scatter(df_district_coords['x'], df_district_coords['y'], 
           color='red', s=5, zorder=5, label='Centroids') # s is size
ax.set_title('118th Congressional Districts and their Geographic Centroids')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.tight_layout()
plt.show()
