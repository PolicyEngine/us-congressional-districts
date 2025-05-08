
import geopandas as gpd
import pandas as pd
import us as us_states_lib

import matplotlib.pyplot as plt
from us_congressional_districts.utils import get_data_directory


# 1. US States Hex Grid GeoJSON URL
# This file contains a custom layout for US states as hexagons
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

fig, ax = plt.subplots(figsize=(15, 9))

# Plot the hexagon outlines from the original GeoDataFrame for context
gdf_hex.plot(ax=ax, facecolor="none", edgecolor="lightgray", linewidth=0.5)

# Plot points at the calculated x, y coordinates
ax.scatter(
    df_coordinates["x"],
    df_coordinates["y"],
    s=50,  # size of markers
    color="red",
    alpha=0.7,
    zorder=5,  # Ensure points are on top
)

# Add state abbreviations as labels next to the points
for index, row in df_coordinates.iterrows():
    ax.text(
        row["x"]
        + 0.01
        * (
            gdf_hex.total_bounds[2] - gdf_hex.total_bounds[0]
        ),  # Slight offset for label
        row["y"],
        row["state_abbr"],
        ha="left",
        va="center",
        fontsize=7,
    )

ax.set_axis_off()  # Turn off the x/y axis lines and labels
ax.set_title("Generated X, Y Coordinates for US State Hexagons")
# plt.show()
