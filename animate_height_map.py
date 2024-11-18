#This is intended to create animations over time for water heights but is currently very buggy and not very usable...

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from glob import glob

# Path to the datasets folder
datasets_folder = "temp"

# Load metadata
metadata_file = os.path.join(datasets_folder, "metadata.csv")
metadata = pd.read_csv(metadata_file)

n=20

# Load site data
site_data = {}
site_files = glob(os.path.join(datasets_folder, "*.csv"))  # All CSV files in datasets folder
site_files = [file for file in site_files if os.path.basename(file) != "metadata.csv"]

for site_file in site_files:
    site_id = os.path.splitext(os.path.basename(site_file))[0]
    data = pd.read_csv(site_file, skiprows=lambda x: x > 0 and (x - 1) % n != 0)
    
    # Combine Date and Time into a datetime column
    data['date_time'] = pd.to_datetime(data['Date']) + pd.to_timedelta(data['Time'], unit='h')
    
    # Save only relevant columns
    try:
        site_data[site_id] = data[['date_time', 'Height']]
    except (IndexError, KeyError):
        continue  # Skip if no data for this site or frame
    

# Define the animation update function
def update_map(frame, ax, metadata, site_data, max_height):
    print("updating animation frame " + str(frame))
    ax.clear()  # Clear the previous frame

    # Set map boundaries and title
    ax.set_xlim(140.5, 150)  # Approximate longitude bounds for Victoria
    ax.set_ylim(-39, -33)    # Approximate latitude bounds for Victoria
    ax.set_title(f"Site Heights at Frame {frame}", fontsize=16)

    # Plot sites
    for _, row in metadata.iterrows():
        site_id = str(row['Site ID'])  # Ensure site ID matches dictionary keys
        lat, lon = row['Latitude'], row['Longitude']
        
        try:
            value = site_data[site_id].iloc[frame]['Height']
        except (IndexError, KeyError):
            continue  # Skip if no data for this site or frame

        # Normalize height value for color mapping
        norm = mcolors.Normalize(vmin=0, vmax=max_height)
        color = cm.viridis(norm(value))

        # Plot the site
        ax.scatter(lon, lat, color=color, s=100, edgecolor='black')

# Get total frames and determine maximum height for color scaling
max_frames = 100 #max(len(data) for data in site_data.values())
max_height = 2.5 #max(data['Height'].max() for data in site_data.values())

print(max_height)

# Initialize the figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create the animation
ani = FuncAnimation(
    fig,
    update_map,
    frames=max_frames,
    fargs=(ax, metadata, site_data, max_height),
    repeat=False
)

# Save the animation as MP4 or GIF
output_file = "outputs/site_height_animation.gif"  # Change to .gif for GIF output
ani.save(output_file, writer='ffmpeg', fps=10)