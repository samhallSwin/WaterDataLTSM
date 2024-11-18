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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import base64
import config
import visualisation_tools

def main():

    check_gpu()

    csv_path = 'datasets/metadata.csv'
    sites =  pd.read_csv(csv_path)

    target_metadata = sites.loc[sites['Site ID'] == config.targetSiteID]
    if not target_metadata.empty:
        print("Data for the Target Site ID:")
        print(target_metadata)
    else:
        print("No data found for the specified Site ID.")
    
    print()
    csv_path = 'datasets/' + str(config.targetSiteID) + '.csv'
    targetData = pd.read_csv(csv_path)

    date_series_target = pd.to_datetime(targetData.pop('Date'), format='%Y-%m-%d')
    time_series_target = pd.to_timedelta(targetData.pop('Time'), unit='h')
    targetData["date_time"] = date_series_target + time_series_target

    input_metadata = []
    InputSiteData = []
    for i in range(len(config.input_Sites)):
        
        input_metadata_temp = sites.loc[sites['Site ID'] == config.input_Sites[i]]
        input_metadata.append(input_metadata_temp)
        if not input_metadata[i].empty:
            print("Data for the input Site ID: ")
            print(input_metadata[i])

        else:
            print("No data found for the specified Site ID.")
        csv_path = 'datasets/' + str(config.input_Sites[i]) + '.csv'
        siteDataTemp = pd.read_csv(csv_path)
        
        date_series_input = pd.to_datetime(siteDataTemp.pop('Date'), format='%Y-%m-%d')
        time_series_input = pd.to_timedelta(siteDataTemp.pop('Time'), unit='h')
        siteDataTemp["date_time"] = date_series_input + time_series_input
        InputSiteData.append(siteDataTemp)
    
    print()
    print('Example data from first site:')
    print(InputSiteData[0].head())
    

    print()
    print("Length of datasets (errors may occur if different):") 
    print(str(len(targetData)))
    for i in range(len(config.input_Sites)):
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
    
    print()
    print('Merged data first 5 entries:')
    print(merged_data.head())

    #visualisation_tools.plotHeights(merged_data)

    if config.makePlots:
        for i in range(len(config.input_Sites)):
            visualisation_tools.plotLocData(InputSiteData[i], "input_" + str(config.input_Sites[i]))
        visualisation_tools.plotLocData(targetData, "target_" + str(config.targetSiteID))

    if config.makeMap:
        print('Generating interactive map - may take a while. Set makePlots = False if not needed.')
        visualisation_tools.interactiveMap()

    print()
    print('Key statistics:')
    print(merged_data.describe().transpose())

    merged_data['date_time'] = pd.to_datetime(merged_data['date_time'])

    merged_data['hour'] = merged_data['date_time'].dt.hour
    merged_data['day'] = merged_data['date_time'].dt.dayofweek

    input_cols = [col for col in merged_data.columns if 'Height' in col and col != 'Height_target']

    target_col = 'Height_target'

    # Normalize the data (scaling the inputs)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = merged_data[input_cols + [target_col]]
    scaled_data = scaler.fit_transform(scaled_data)

    print('merged_data shape = ' + str(merged_data[input_cols + [target_col]].shape))
    print('scaled_data shape = ' + str(scaled_data.shape))

    # Split the data into features and target
    X = scaled_data[:, :-1]  # Inputs (all Height columns)
    y = scaled_data[:, -1]   # Target (Height_target)

    X_seq, y_seq = create_sequences(X, y, timesteps=config.timesteps, n_future=config.time_ahead)

    print("X_seq shape = " + str(X_seq.shape))
    print("y_seq shape = " + str(y_seq.shape))
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    print("y_train shape = " + str(y_train.shape))
    print("y_test shape = " + str(y_test.shape))

    model = Sequential()
    model.add(LSTM(units=config.units, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=config.learning_rate), loss='mean_squared_error')
    model.summary()

    history = model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size, validation_data=(X_test, y_test))

    print(history)

    output_dir = 'outputs/models/'
    output_path = os.path.join(output_dir, f'{config.experiment_name}.h5')

    os.makedirs(output_dir, exist_ok=True)

    model.save(output_path)

    print(f"Model saved to {output_path}")

    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

    predictions = model.predict(X_test)
    
    placeholder = 0

    #Reshape data for inverse scaling for plotting - awkward fix for reshaping due to LSTM
    X_slice = X_test[:, -1, :]
    pred_slice = predictions[:,-1].reshape(-1, 1) 

    placeholder = 0
    shifted_pred = np.full_like(pred_slice, placeholder)
    shifted_pred[:-config.time_ahead] = pred_slice[config.time_ahead:]

    y_slice = y_test.reshape(-1, 1) 
    predictions_combined = np.concatenate([X_slice, shifted_pred], axis=1)
    Y_combined = np.concatenate([X_slice, y_slice], axis=1)
    
    predictions_original_scale = scaler.inverse_transform(predictions_combined)[:, -1]

    y_test_original_scale = scaler.inverse_transform(Y_combined)[:, -1]

    visualisation_tools.plotPredicted(y_test_original_scale, predictions_original_scale, len(predictions_original_scale))


    #n = config.time_ahead - 1
    #visualisation_tools.plotPredicted(y_test[:,n], predictions[:,n])

def create_sequences(X, y, timesteps, n_future):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps - n_future + 1):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps + n_future - 1])
    return np.array(X_seq), np.array(y_seq)

def check_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    main()