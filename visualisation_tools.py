import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import folium
import seaborn as sns
import config
import os

def interactiveMap2():
    csv_path = 'datasets/metadata.csv'
    df = pd.read_csv(csv_path)
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=8)

    # Add markers to the map
    for idx, row in df.iterrows():
        popup_text = f"Site ID: {row['Site ID']}<br>Location: {row['Location']}"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_text,
            tooltip=row['Basin']
        ).add_to(m)

    # Save to an HTML file
    m.save("outputs/site_locations_map.html")

def interactiveMap():
    # Load metadata
    csv_path = 'datasets/metadata.csv'
    df = pd.read_csv(csv_path)
    
    # Create a folium map
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=8)
    
    # Add markers with graphs in popups
    for idx, row in df.iterrows():
        site_id = row['Site ID']
        graph_base64 = generate_graph(site_id)
        if graph_base64 is not None:
            graph_html = f'''
            <div style="width:600px; height:400px; overflow:auto;">
                <img src="data:image/png;base64,{graph_base64}" style="width:100%;"/>
            </div>
            '''
        else:
            graph_html = "No valid data available for this site."
        
        popup_text = f"""
        <b>Site ID:</b> {site_id}<br>
        <b>Location:</b> {row['Location']}<br>
        {graph_html}
        """
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_text, max_width=650),  # Increased popup width
            tooltip=row['Basin']
        ).add_to(m)

    # Save to an HTML file
    output_path = "outputs/site_locations_map.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Map saved to {output_path}")

def generate_graph(site_id):
    """Generate a graph for the given Site ID."""
    data_path = f"datasets/{site_id}.csv"
    if not os.path.exists(data_path):
        return None  # Skip if the dataset does not exist

    # Load dataset
    df = pd.read_csv(data_path)

    # Combine Date and Time into a single datetime column
    df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].astype(str) + ':00:00')

    # Check which columns are available for plotting
    fields_to_plot = []
    if 'Flow' in df.columns:
        fields_to_plot.append(('Flow', 'blue'))
    if 'Height' in df.columns:
        fields_to_plot.append(('Height', 'green'))
    if 'Rainfall' in df.columns:
        fields_to_plot.append(('Rainfall', 'orange'))

    if not fields_to_plot:
        return None  # No valid fields to plot

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))  # Larger figure size

    # Primary y-axis for Flow
    ax2 = ax1.twinx()  # Secondary axis for Rainfall
    ax3 = ax1.twinx()  # Tertiary axis for Height

    # Offset the tertiary axis to the left
    ax3.spines['right'].set_position(('outward', 60))
    ax3.spines['right'].set_visible(True)

    # Plot fields with appropriate axes
    if 'Flow' in df.columns:
        ax1.plot(df['date_time'], df['Flow'], label='Flow', color='blue')
        ax1.set_ylabel('Flow', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
    if 'Rainfall' in df.columns:
        ax2.plot(df['date_time'], df['Rainfall'], label='Rainfall', color='orange')
        ax2.set_ylabel('Rainfall', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
    if 'Height' in df.columns:
        ax3.plot(df['date_time'], df['Height'], label='Height', color='green', linestyle='dashed')
        ax3.set_ylabel('Height', fontsize=12, color='green')
        ax3.tick_params(axis='y', labelcolor='green')

    # Set the x-axis label and title
    ax1.set_xlabel('Date-Time', fontsize=12)
    plt.title(f'Site ID: {site_id} - Available Data Over Time', fontsize=14)

    # Add grid lines
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Legends for all axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=10)

    # Adjust layout to fit the plot
    fig.tight_layout()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')  # Ensure no cropping
    plt.close()
    buffer.seek(0)

    # Convert to base64 string
    graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return graph_base64


def plotLocData(data, id):
    # Set the date_time column as the index if it’s not already
    data = data.set_index('date_time')

    # Specify columns to plot
    plot_cols = ['Rainfall', 'Flow', 'Height']
    
    # Create subplots
    fig, axes = plt.subplots(nrows=len(plot_cols), ncols=1, figsize=(10, 8), sharex=True)

    # Filter the plot columns that are actually present in the dataset
    present_cols = [col for col in plot_cols if col in data.columns]

    # Plot each feature that is present
    for i, col in enumerate(present_cols):
        data[col].plot(ax=axes[i], title=col)
        axes[i].set_ylabel(col)

    # Adjust axes to match the number of available columns
    if len(present_cols) < len(plot_cols):
        # Hide extra axes if there are fewer columns than expected
        for i in range(len(present_cols), len(plot_cols)):
            axes[i].set_visible(False)

    # Set the common x-label for time
    plt.xlabel('Date Time')

    # Adjust layout and save plot to disk
    plt.tight_layout()
    plt.savefig(f"outputs/time_series_plot_{id}.png")
    plt.close(fig)
    print(f"Plot saved as 'time_series_plot_{id}.png'")

def plotPredicted(y_test_actual, predictions_actual, length):

    print("Saving predicted vs actual figure")
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the actual vs predicted values
    plt.plot(y_test_actual[:length], label='Actual Height_target', color='blue', linewidth=2)
    plt.plot(predictions_actual[:length], label='Predicted Height_target', color='red', linestyle='--', linewidth=2)

    # Add title and labels
    plt.title('Comparison of Actual vs Predicted Height_target', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Height_target', fontsize=12)

    # Add a legend
    plt.legend()

    # Save the plot to a file
    output_dir = 'outputs/'
    output_path = os.path.join(output_dir, f'{config.experiment_name}_comparison_plot.png')
    plt.savefig(output_path)  # Change the file path if needed

def plotHeights(data):

    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame(data)

    # Convert 'date_time' column to datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Create a plot of each 'Height' column
    plt.figure(figsize=(10, 6))

    plt.plot(df['date_time'], df['Height'], label='Height', color='blue', linewidth=2)
    plt.plot(df['date_time'], df['Height_target'], label='Height_target', color='red', linewidth=2)


    # Add labels and a title
    plt.xlabel('Date and Time')
    plt.ylabel('Height')
    plt.title('Height Data Over Time')

    # Add a legend
    plt.legend()

    # Save the plot as an image file
    plt.savefig('height_plot.png', dpi=300)