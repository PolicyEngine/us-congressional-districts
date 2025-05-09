import geopandas as gpd
import pandas as pd
import us as us_states_lib

from us_congressional_districts.utils import get_data_directory


# 1. US States Hex Grid GeoJSON URL
# This file contains a custom layout for US states as hexagons
# Article: https://python-graph-gallery.com/hexbin-map-from-geojson-python
hex_grid_url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/us_states_hexgrid.geojson.json"

# 2. Load the Hex Grid using geopandas
try:
    gdf_hex = gpd.read_file(hex_grid_url)
except Exception as e:
    print(f"Error loading GeoJSON: {e}")
    if "geometry" not in gdf_hex.columns:
        print("GeoJSON did not load correctly or is missing geometry.")

# 3. Calculate Centroid (x, y) Coordinates for each Hexagon
# The 'geometry' column in gdf_hex contains the polygon for each state's hexagon.
# We calculate the centroid of each polygon to get its center point.
gdf_hex["centroid"] = gdf_hex["geometry"].centroid
gdf_hex["x"] = gdf_hex["centroid"].x
gdf_hex["y"] = gdf_hex["centroid"].y

# 4. Create Mappings for State Identifiers using the 'us' library
# This helps us include FIPS codes and ensure consistency.
fips_to_abbr = us_states_lib.states.mapping("fips", "abbr")
abbr_to_fips = {
    v: k for k, v in fips_to_abbr.items()
}  # Inverse mapping for FIPS lookup
abbr_to_name = us_states_lib.states.mapping("abbr", "name")  # For full state names

# 5. Prepare the final DataFrame with FIPS, Abbreviation, Name, x, and y
state_coordinates_list = []

# Iterate through the rows of the GeoDataFrame to extract the information
for index, row in gdf_hex.iterrows():
    # 'iso3166_2' usually holds the state abbreviation in this specific GeoJSON
    state_abbr_from_geojson = row.get("iso3166_2")

    if state_abbr_from_geojson:
        fips_code = abbr_to_fips.get(state_abbr_from_geojson)
        state_name = abbr_to_name.get(state_abbr_from_geojson)

        state_coordinates_list.append(
            {
                "fips": fips_code,
                "state_abbr": state_abbr_from_geojson,
                "state_name": state_name,
                "x": row["x"],
                "y": row["y"],
            }
        )
    else:
        # Handle cases where 'iso3166_2' might be missing or different
        # You might want to inspect 'row' to find the correct identifier if this happens
        print(
            f"Warning: Missing 'iso3166_2' or identifier for a row: {row.get('name', 'N/A')}"
        )

df_coordinates = pd.DataFrame(state_coordinates_list)
df_coordinates.sort_values("fips", inplace=True)
df_coordinates[["GEO_ID"]] = "0400000US" + df_coordinates[["fips"]]
df_coordinates.dropna(subset="fips", inplace=True)

csv_file = get_data_directory() / "input" / "geographies" / "states.csv"
output_cols = ["GEO_ID", "x", "y"]
df_coordinates[output_cols].to_csv(csv_file, index=False)


# --- Optional: Example of how you might use these coordinates for a basic plot ---

import plotly.graph_objects as go
import numpy as np

# Create figure
fig = go.Figure()

# Add hexagon outlines from the GeoDataFrame
for idx, row in gdf_hex.iterrows():
    # Extract geometry coordinates and convert to lists
    x, y = row.geometry.exterior.xy
    x_list = list(x)
    y_list = list(y)
    
    fig.add_trace(
        go.Scatter(
            x=x_list,
            y=y_list,
            mode='lines',
            line=dict(color='lightgray', width=0.5),
            fill='none',
            showlegend=False,
            hoverinfo='skip'
        )
    )

# Add points at the x, y coordinates stored in the dataframe
fig.add_trace(
    go.Scatter(
        x=gdf_hex["x"].tolist(),  # Convert to list to ensure compatibility
        y=gdf_hex["y"].tolist(),  # Convert to list to ensure compatibility
        mode="markers",
        marker=dict(
            size=12,
            color="red",
            opacity=0.7
        ),
        showlegend=False,
        hoverinfo='text',
        hovertext=gdf_hex["label"].tolist()  # Convert to list
    )
)

# Add state abbreviation labels
fig.add_trace(
    go.Scatter(
        x=(gdf_hex["x"] + 0.5).tolist(),  # Convert to list
        y=gdf_hex["y"].tolist(),  # Convert to list
        mode="text",
        text=gdf_hex["label"].tolist(),  # Convert to list
        textposition="middle right",
        textfont=dict(
            size=10,
        ),
        showlegend=False,
        hoverinfo='skip'
    )
)

# Update layout
fig.update_layout(
    title="US State Hexagon Map",
    width=1200,
    height=720,
    plot_bgcolor='white',
    margin=dict(l=0, r=0, t=50, b=0),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        showline=False,
        scaleanchor="x",  # This ensures the aspect ratio is 1:1
        scaleratio=1
    )
)

fig.show()
