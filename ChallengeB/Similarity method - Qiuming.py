import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None, low_memory=False)
    numeric_columns = list(range(1, data.shape[1]))
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data.dropna(inplace=True)
    return data

def prepare_data(data):
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 1:].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_scaled)
    X_scaled = X_scaled.astype('float32')
    y = y.astype('float32')
    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))
    return X_scaled, y

def create_autoencoder(input_shape, encoding_dim):
    input_data = Input(shape=(input_shape,))
    encoded = Dense(64, activation='tanh')(input_data)
    encoded = Dense(encoding_dim, activation='tanh')(encoded)
    decoded = Dense(64, activation='tanh')(encoded)
    decoded = Dense(input_shape, activation='linear')(decoded)
    autoencoder = Model(input_data, decoded)
    encoder = Model(input_data, encoded)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(loss='mean_squared_error', optimizer=optimizer)
    return autoencoder, encoder

def calculate_similarity_for_target_gene(target_gene_index, encoded_genes):
    similarity_vector = cosine_similarity(encoded_genes[target_gene_index].reshape(1, -1), encoded_genes).flatten()
    return similarity_vector

def save_similarity_matrix(similarity_matrix, gene_names, target_genes, file_name="similarity_matrix.csv"):
    similarity_matrix_df = pd.DataFrame(similarity_matrix.T, columns=gene_names, index=target_genes)
    similarity_matrix_df.index.name = 'Target Gene'
    similarity_matrix_df.to_csv(file_name)

if __name__ == "__main__":
    file_path = "Challenge_GIN_release_profile_17804library_181query.txt"
    data = load_and_preprocess_data(file_path)
    X_scaled, y = prepare_data(data)

    encoding_dim = 32
    autoencoder, encoder = create_autoencoder(X_scaled.shape[1], encoding_dim)
    print("Training the autoencoder...")
    history = autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

    encoded_genes = encoder.predict(X_scaled)

    target_genes = [193, 19, 81, 48, 63, 205, 191, 24, 60, 186, 75, 57, 185, 125, 114, 22, 218, 0, 178, 97, 126, 54, 20, 142, 12, 182, 159, 77, 15, 82, 183, 206, 138, 80, 215, 2, 149, 32, 188, 7]

    similarity_matrix = np.zeros((encoded_genes.shape[0], len(target_genes)))

    for idx, target_gene_index in enumerate(target_genes):
        similarity_vector = calculate_similarity_for_target_gene(target_gene_index, encoded_genes)
        similarity_matrix[:, idx] = similarity_vector

    gene_names = data.iloc[:, 0].values.tolist()
    save_similarity_matrix(similarity_matrix, gene_names, target_genes, "similarity_matrix.csv")

    # Load the similarity matrix into a DataFrame
df = pd.read_csv('similarity_matrix.csv', index_col=0)

# Transpose the DataFrame
df_transposed = df.transpose()

# Save the transposed DataFrame to a new file
df_transposed.to_csv('similarity_matrix_transposed.csv')

