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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import base64
import config
import visualisation_tools
import logging

def main():

    check_gpu()

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[
        logging.StreamHandler(),  # Output to terminal
        logging.FileHandler(f'logs/{config.experiment_name}.log'),  # Save logs to a file
        ],
    )
    

    csv_path = 'datasets/metadata.csv'
    sites =  pd.read_csv(csv_path)

    # Load and preprocess the target site data
    target_metadata = sites.loc[sites['Site ID'] == config.targetSiteID]
    if not target_metadata.empty:
        logging.info("Data for the Target Site ID:")
        logging.info(target_metadata)
    else:
        logging.error("No data found for the specified Target Site ID.")
        exit(1)
    
   
    csv_path = f'datasets/{config.targetSiteID}.csv'
    targetData = preprocess_site_data(csv_path, config.target_fields)


    # Load and preprocess the input site data
    InputSiteData = []
    for site_config in config.input_Sites:
        site_id = site_config["site_id"]
        csv_path = f'datasets/{site_id}.csv'
        site_data = preprocess_site_data(csv_path, site_config["fields"])
        InputSiteData.append(site_data)


    # Merge input site data
    merged_Input_data = InputSiteData[0][['date_time']].copy()
    for i, site_data in enumerate(InputSiteData):
        site_id = config.input_Sites[i]["site_id"]
        fields_to_merge = ['date_time'] + [field.capitalize() for field in config.input_Sites[i]["fields"]]
        merged_Input_data = pd.merge(
            merged_Input_data,
            site_data[fields_to_merge],
            on='date_time',
            suffixes=('', f'_{site_id}'),
            how='outer'
        )

    # Merge with target data
    target_fields_to_merge = ['date_time'] + [field.capitalize() for field in config.target_fields]
    merged_data = pd.merge(
        merged_Input_data,
        targetData[target_fields_to_merge],
        on='date_time',
        suffixes=('', '_target'),
        how='outer'
    )

    logging.info('Merged data first 5 entries:')
    logging.info(merged_data.head())

    #visualisation_tools.plotHeights(merged_data)

    # Generate plots if configured
    if config.makePlots:
        for i, site_data in enumerate(InputSiteData):
            site_id = config.input_Sites[i]["site_id"]
            visualisation_tools.plotLocData(site_data, f"input_{site_id}")
        visualisation_tools.plotLocData(targetData, f"target_{config.targetSiteID}")

    if config.makeMap:
        logging.info('Generating interactive map - may take a while. Set makeMap=False if not needed.')
        visualisation_tools.interactiveMap()

    logging.info('Key statistics:')
    logging.info(merged_data.describe().transpose())

    # Feature engineering
    merged_data['date_time'] = pd.to_datetime(merged_data['date_time'])
    merged_data['hour'] = merged_data['date_time'].dt.hour
    merged_data['day'] = merged_data['date_time'].dt.dayofweek

    # Define input and target columns
    input_cols = [col for col in merged_data.columns if any(f.capitalize() in col for f in config.input_Sites[0]["fields"])]
    target_cols = [col for col in merged_data.columns if col.endswith('_target')]

    print(f'target cols = {target_cols}')

    # Step 1: Split the data BEFORE scaling
    train_size = int(len(merged_data) * 0.8)  # Example: 80% training, 20% testing
    train_data = merged_data.iloc[:train_size]
    test_data = merged_data.iloc[train_size:]

    # Apply MinMaxScaler
    #scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data[input_cols + target_cols])  # Fit only on training data


    # Scale data
    train_scaled = scaler.transform(train_data[input_cols + target_cols])
    test_scaled = scaler.transform(test_data[input_cols + target_cols])

    # Extract inputs (X) and targets (y)
    X_train = train_scaled[:, :-len(target_cols)]
    y_train = train_scaled[:, -len(target_cols):]
    X_test = test_scaled[:, :-len(target_cols)]
    y_test = test_scaled[:, -len(target_cols):]

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, config.timesteps, config.time_ahead)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, config.timesteps, config.time_ahead)

    # Step 6: Train the model
    model, history = train_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, config, target_cols)

    if config.makePlots:
        visualisation_tools.plot_loss(history, config.experiment_name)

    logging.info(history)

    # Save the trained model
    output_dir = 'outputs/models/'
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, f'{config.experiment_name}.h5'))
    logging.info(f"Model saved to {output_dir}")

    # Evaluate the model
    loss = model.evaluate(X_test_seq, y_test_seq)
    logging.info(f"Test Loss (aggregated across outputs): {loss}")

    # Make predictions
    predictions = model.predict(X_test_seq)

    #visualisation_tools.log_evaluation_metrics(y_test, predictions)

    ## Plot predictions if enabled
    #if config.makePlots:
    #    visualisation_tools.plot_predictions(y_test, predictions, config.experiment_name)

    # Inverse scale predictions and true values
    X_slice = X_test_seq[:, -1, :]
    predictions_combined = np.concatenate([X_slice, predictions], axis=1)
    Y_combined = np.concatenate([X_slice, y_test_seq], axis=1)

    # Apply inverse scaling
    predictions_original_scale = scaler.inverse_transform(predictions_combined)[:, -len(predictions[0]):]
    y_test_original_scale = scaler.inverse_transform(Y_combined)[:, -len(y_test_seq[0]):]

    # Save predictions to CSV
    if config.predictions2csv:
        save_predictions_to_csv(y_test_original_scale, predictions_original_scale, config.experiment_name)

    # Plot each output separately
    for i, target_field in enumerate(config.target_fields):
        visualisation_tools.plotPredicted(
            y_test_original_scale[:, i],  # True values for this output
            predictions_original_scale[:, i],  # Predicted values for this output
            length =  len(predictions_original_scale[:, i]),
            label=target_field.capitalize(),  # Use target field name for the label
            output_name=target_field
    )


    #n = config.time_ahead - 1
    #visualisation_tools.plotPredicted(y_test[:,n], predictions[:,n])

def weighted_mse(y_true, y_pred):
    weights = tf.where(y_true > 0.8, 2.0, 1.0)  # Higher weight for extreme values
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

def build_lstm_model(input_shape, units, num_outputs, learning_rate):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout for regularization
    model.add(BatchNormalization())  # Normalize outputs
    model.add(LSTM(units=units // 2, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_outputs, activation='linear'))  # Multi-output regression
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='weighted_mse')
    return model

def train_model(X_train, y_train, X_test, y_test, config, target_cols):
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=config.units,
        num_outputs=len(target_cols),
        learning_rate=config.learning_rate
    )
    model.summary()

    # Compile the model with the custom loss
    model.compile(optimizer=Adam(learning_rate=config.learning_rate), loss=weighted_mse)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{config.experiment_name}')

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    return model, history


def create_sequences(X, y, timesteps, n_future):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps - n_future + 1):
        X_seq.append(X[i:i + timesteps])  # Past `timesteps` for input
        y_seq.append(y[i + timesteps + n_future - 1])  # Target `n_future` steps ahead
    return np.array(X_seq), np.array(y_seq)


def check_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if not physical_devices:
        logging.info("Warning: No GPU found. The program will use the CPU instead (SLOW).")
    else:
        logging.info("Num GPUs Available: ", len(physical_devices))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

def save_predictions_to_csv(y_true, y_pred, experiment_name):
    """
    Saves true and predicted values for all outputs to a CSV file.

    Args:
        y_true: True target values (multi-output).
        y_pred: Predicted target values (multi-output).
        experiment_name: Name of the experiment for file identification.
    """
    output_dir = 'outputs/predictions'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"predictions_{experiment_name}.csv")

    # Create a DataFrame for predictions
    data = {}
    for i, field in enumerate(config.target_fields):
        data[f'True_{field.capitalize()}'] = y_true[:, i]
        data[f'Predicted_{field.capitalize()}'] = y_pred[:, i]

    results_df = pd.DataFrame(data)

    # Save to CSV
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Predictions saved to {csv_path}")

def preprocess_site_data(csv_path, fields):
    site_data = pd.read_csv(csv_path)
    date_series = pd.to_datetime(site_data.pop('Date'), format='%Y-%m-%d')
    time_series = pd.to_timedelta(site_data.pop('Time'), unit='h')
    site_data["date_time"] = date_series + time_series

    fields = ['date_time'] + [field.capitalize() for field in fields if field.capitalize() in site_data.columns]
    return site_data[fields]

if __name__ == "__main__":
    main()