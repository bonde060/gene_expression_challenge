from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr


def load_and_preprocess_data(file_path):
    data = pd.read_table("Challenge_GIN_release_profile_17804library_181query.txt",
                         delimiter="\t", header=0, index_col=0)
    # take rows that are present in the columns
    # print(data.columns)
    X = data.loc[data.columns]
    # print(X)
    # transpose is the output
    Y = data.T
    # print(Y.shape)
    # print(Y)
    return data, X, Y


def prepare_data(data, X):
    # Prepare the input and output data
    # X = data.iloc[:, 1:].values

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
    model.add(Dense(17804))

    # Use a smaller learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def save_predicted_interactions(predictions, file_name="predicted_interactions.xlsx"):
    # Save the predicted interactions to an Excel file
    predicted_interactions_df = pd.DataFrame(predictions).T
    predicted_interactions_df.to_excel(file_name, index=False)


if __name__ == "__main__":
    file_path = "Challenge_GIN_release_profile_17804library_181query.txt"
    data, X, Y = load_and_preprocess_data(file_path)
    X_scaled = prepare_data(data, X)
    model = create_model(X_scaled.shape[1])
    
    print(type(X_scaled), type(Y))

    # Train the model
    print("Training the model...")
    history = model.fit(X_scaled, Y, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

    # Make predictions for all genes
    predicted_interactions = model.predict(X_scaled[:1])
    print(predicted_interactions)
    print(X_scaled[:1])
    corr, pval = pearsonr(predicted_interactions[0], Y.iloc[0])

    print(corr)

    # Save the predicted interactions to an Excel file
    save_predicted_interactions(predicted_interactions, "all_predicted_interactions.xlsx")
    print("Predicted interactions for all genes saved to 'all_predicted_interactions.xlsx'")

    # Get the interactions for the given index
    # index = 6000
    # interactions = predicted_interactions[index, :]
    # print(f"Interactions for gene {index}: {interactions}")







