from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf


def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path, sep="\t", header=None, low_memory=False)

    # Preprocess the data
    numeric_columns = list(range(2, data.shape[1]))
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data.dropna(inplace=True)
    return data


def prepare_data(data):
    # Prepare the input and output data
    X = data.iloc[:, 1:].values

    # Scale the input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalize the input data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_scaled)
    X_scaled = X_scaled.astype('float32')

    # Check for NaN or infinite values
    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))

    return X_scaled


def create_model(input_shape):
    # Create the neural network model
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(input_shape))

    # Use a smaller learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def save_predicted_interactions(predictions, file_name="predicted_interactions.xlsx"):
    # Save the predicted interactions to an Excel file
    predicted_interactions_df = pd.DataFrame(predictions)
    predicted_interactions_df.to_excel(file_name, index=False)


if __name__ == "__main__":
    file_path = "Challenge_GIN_release_profile_17804library_181query.txt"
    data = load_and_preprocess_data(file_path)
    X_scaled = prepare_data(data)
    model = create_model(X_scaled.shape[1])

    # Train the model
    print("Training the model...")
    history = model.fit(X_scaled, X_scaled, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

    # Make predictions for all genes
    predicted_interactions = model.predict(X_scaled)

    # Save the predicted interactions to an Excel file
    save_predicted_interactions(predicted_interactions, "all_predicted_interactions.xlsx")
    print("Predicted interactions for all genes saved to 'all_predicted_interactions.xlsx'")

    # Get the interactions for the given index
    index = 6000
    interactions = predicted_interactions[index, :]
    print(f"Interactions for gene {index}: {interactions}")







