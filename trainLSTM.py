import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import folium
import seaborn as sns
import tensorflow as tf
from io import BytesIO
import base64


def main():

    makePlots = True

    targetSiteID = 407254
    input_Sites = [407246, 407255, 407229]

    input_width = 24
    label_width = 1
    shift = 24

    csv_path = 'datasets/metadata.csv'
    sites =  pd.read_csv(csv_path)

    target_metadata = sites.loc[sites['Site ID'] == targetSiteID]
    if not target_metadata.empty:
        print("Data for the Target Site ID:")
        print(target_metadata)
    else:
        print("No data found for the specified Site ID.")
    
    print()
    csv_path = 'datasets/' + str(targetSiteID) + '.csv'
    targetData = pd.read_csv(csv_path)

    date_series_target = pd.to_datetime(targetData.pop('Date'), format='%Y-%m-%d')
    time_series_target = pd.to_timedelta(targetData.pop('Time'), unit='h')
    targetData["date_time"] = date_series_target + time_series_target

    input_metadata = []
    InputSiteData = []
    for i in range(len(input_Sites)):
        
        input_metadata_temp = sites.loc[sites['Site ID'] == input_Sites[i]]
        input_metadata.append(input_metadata_temp)
        if not input_metadata[i].empty:
            print("Data for the input Site ID: ")
            print(input_metadata[i])

        else:
            print("No data found for the specified Site ID.")
        csv_path = 'datasets/' + str(input_Sites[i]) + '.csv'
        siteDataTemp = pd.read_csv(csv_path)
        
        date_series_input = pd.to_datetime(siteDataTemp.pop('Date'), format='%Y-%m-%d')
        time_series_input = pd.to_timedelta(siteDataTemp.pop('Time'), unit='h')
        siteDataTemp["date_time"] = date_series_input + time_series_input
        InputSiteData.append(siteDataTemp)
    

    print(InputSiteData[1].head())
    

    print()
    print("Length of datasets (errors may occur if different):") 
    print(str(len(targetData)))
    for i in range(len(input_Sites)):
        print(str(len(InputSiteData[i])))

    #now merge the data into one object for ease of use
    merged_Input_data = InputSiteData[0][['date_time', 'Flow', 'Height']]

    # Iteratively merge all datasets on 'date_time'
    for i in range(1, len(InputSiteData)):
        # Merge each dataset on 'date_time', creating unique column names using suffixes
        merged_Input_data = pd.merge(merged_Input_data, 
                            InputSiteData[i][['date_time', 'Flow', 'Height']], 
                            on='date_time', 
                            suffixes=('', f'_{i+1}'), how='outer')


    merged_data = pd.merge(merged_Input_data, 
                        targetData[['date_time', 'Flow', 'Height']], 
                        on='date_time', 
                        suffixes=('', '_target'), how='outer')
        
    print(merged_data.head())

    if makePlots:
        for i in range(len(input_Sites)):
            plotLocData(InputSiteData[i], "input_" + str(input_Sites[i]))

        plotLocData(InputSiteData[i], "target_" + str(targetSiteID))
        print('Generating interactive map - may take a while. Set makePlots = False if not needed.')
        interactiveMap()

    column_indices = {name: i for i, name in enumerate(merged_data.columns)}

    #split train, test and val sets (no shuffle due to time)
    n = len(merged_data)
    train_df = merged_data[0:int(n*0.7)]
    val_df = merged_data[int(n*0.7):int(n*0.9)]
    test_df = merged_data[int(n*0.9):]

    num_features = merged_data.shape[1]
    print(merged_data.shape)

    print()
    print('Key statistics:')
    print(merged_data.describe().transpose())

    #normalise
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    if makePlots:
        df_std = (merged_data - train_mean) / train_std
        df_std = df_std.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(merged_data.keys(), rotation=90)
        plt.savefig(f"outputs/standardd.png")

    w1 = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift,
                     label_columns=['Height_target'],train_df=train_df, val_df=val_df, test_df=test_df)
    print(w1)

    WindowGenerator.split_window = split_window 

    # Stack three slices, the length of the total window.
    example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                            np.array(train_df[100:100+w1.total_window_size]),
                            np.array(train_df[200:200+w1.total_window_size])])

    example_inputs, example_labels = w1.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'Labels shape: {example_labels.shape}')

    if makePlots:
        WindowGenerator.windowPlot = windowPlot
        w1.example = example_inputs, example_labels
        w1.windowPlot()

    WindowGenerator.make_dataset = make_dataset
    
    WindowGenerator.train = train
    WindowGenerator.val = val
    WindowGenerator.test = test
    WindowGenerator.example = example
    w1.train.element_spec


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

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

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

def windowPlot(self, model=None, plot_col='Height_target', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')
  plt.savefig(f"outputs/windows.png")

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

    # Plot the available fields with dual y-axes for different scales
    fig, ax1 = plt.subplots(figsize=(12, 8))  # Larger figure size
    ax2 = ax1.twinx()  # Create a secondary y-axis

    for field, color in fields_to_plot:
        if field == 'Flow' or field == 'Rainfall':
            ax1.plot(df['date_time'], df[field], label=field, color=color)
        elif field == 'Height':
            ax2.plot(df['date_time'], df[field], label=field, color=color, linestyle='dashed')

    # Customize axes
    ax1.set_xlabel('Date-Time', fontsize=12)
    ax1.set_ylabel('Flow/Rainfall', fontsize=12, color='blue')
    ax2.set_ylabel('Height', fontsize=12, color='green')

    # Title and legend
    plt.title(f'Site ID: {site_id} - Available Data Over Time', fontsize=14)
    fig.tight_layout()  # Adjust layout to avoid clipping
    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

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

if __name__ == "__main__":
    main()