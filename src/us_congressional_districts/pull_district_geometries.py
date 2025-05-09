import geopandas as gpd
import requests
import io

from us_congressional_districts.utils import get_data_directory


YEAR = 2023
geoid_col_shp = "GEOIDFQ" if YEAR == 2023 else "AFFGEOID"  # the identifier for the geography

# 1. Define the URL for the Congressional District shapefile ZIP
zip_file_url = f"https://www2.census.gov/geo/tiger/GENZ{YEAR}/shp/cb_{YEAR}_us_cd118_5m.zip"

# Column names we expect in the shapefile (based on typical Census CD files)
# GEOID column usually contains STATEFP + CD118FP (e.g., "0101" for AL Dist 1)
try:
    response = requests.get(zip_file_url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

    zip_file_bytes = io.BytesIO(response.content)
    gdf_cd = gpd.read_file(zip_file_bytes)

except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")
except Exception as e:
    print(f"Error reading shapefile from downloaded ZIP content: {e}")

print(f"\nColumns in the GeoDataFrame: {gdf_cd.columns.tolist()}")
print(f"CRS of the GeoDataFrame: {gdf_cd.crs}")
print(gdf_cd.head())

if geoid_col_shp not in gdf_cd.columns:
    print(f"Error: Expected GEOID column '{geoid_col_shp}' not found in the shapefile.")
    print("Please inspect the columns above and adjust 'geoid_col_shp'.")

# 5. Calculate Centroids
# The .centroid attribute gives the geometric center.
# For geographic data (lat/lon), these will be lat/lon coordinates.
# If you need centroids for a projected map display, you might reproject first.
# For now, we'll get the geographic centroids.
gdf_cd["geometry_centroid"] = gdf_cd.geometry.centroid
gdf_cd["x"] = gdf_cd["geometry_centroid"].x  # Typically Longitude
gdf_cd["y"] = gdf_cd["geometry_centroid"].y  # Typically Latitude

# 6. Create a DataFrame with the desired information
output_columns = [geoid_col_shp, "x", "y"]
df_district_coords = gdf_cd[output_columns].copy()
df_district_coords.rename(columns={geoid_col_shp: "GEO_ID"}, inplace=True)

# List of non-voting delegate district GEOIDs
non_voting_geoids = [
    '5001800US1198',  # District of Columbia
    '5001800US6098',  # American Samoa
    '5001800US6698',  # Guam
    '5001800US6998',  # Northern Mariana Islands
    '5001800US7298',  # Puerto Rico
    '5001800US7898'   # U.S. Virgin Islands
]

# Filter them out
df_district_coords = df_district_coords[~df_district_coords['GEO_ID'].isin(non_voting_geoids)]

df_district_coords.sort_values("GEO_ID", inplace=True)

df_district_coords.shape


# Save to the inputs directory
csv_output_path = get_data_directory() / "input" / "geographies" / "districts.csv"
df_district_coords.to_csv(csv_output_path, index=False)


# --- Optional: Basic Plot of Districts and their Centroids ---

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a plotly figure
fig = make_subplots(rows=1, cols=1)

# Add the GeoDataFrame as a choropleth map
# We need to convert the GeoDataFrame to a GeoJSON format for Plotly
for idx, row in gdf_cd.iterrows():
    fig.add_trace(
        go.Choropleth(
            geojson=row.geometry.__geo_interface__,
            locations=[idx],
            z=[1],  # Just using a constant value for coloring
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            showscale=False,
            marker=dict(
                line=dict(color="grey", width=0.5),
                opacity=0.7,
            ),
            hoverinfo="skip",
        )
    )

# Add the centroids as scatter points
fig.add_trace(
    go.Scatter(
        x=df_district_coords["x"],
        y=df_district_coords["y"],
        mode="markers",
        marker=dict(color="red", size=5),
        name="Centroids",
    )
)

# Update layout
fig.update_layout(
    title="118th Congressional Districts and their Geographic Centroids",
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    legend=dict(x=0.01, y=0.99),
    width=1200,
    height=800,
    margin=dict(l=50, r=50, b=50, t=50),
    geo=dict(
        projection_type="albers usa",
        showland=True,
        landcolor="white",
        showocean=True,
        oceancolor="lightblue",
    )
)

# For a better geo visualization, we can use update_geos
fig.update_geos(
    fitbounds="locations",
    visible=False,
)

fig.show()
