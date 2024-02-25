import pandas as pd
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import precision_score, recall_score, mean_squared_error

# Set the random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load the click stream data
clicks_df = pd.read_csv('yoochoose-clicks.dat', header=None, names=['SessionId', 'Timestamp', 'ItemId', 'Category'])
clicks_df = clicks_df.drop('Timestamp', axis=1)

# Load the buy event data
buys_df = pd.read_csv('yoochoose-buys.dat', header=None,
                      names=['SessionId', 'Timestamp', 'ItemId', 'Price', 'Quantity'])
buys_df = buys_df.drop(['Timestamp', 'Price', 'Quantity'], axis=1)

# Load the category tree data
categories_df = pd.read_csv('yoochoose-category.dat', header=None, names=['CategoryId', 'CategoryName'])
categories_df = categories_df.set_index('CategoryId')

# Preprocess the data
clicks_df = clicks_df[clicks_df['ItemId'].isin(buys_df['ItemId'].unique())]  # Filter out items that were not bought
clicks_df['ItemId'] = clicks_df['ItemId'].astype(str)
clicks_df['SessionId'] = clicks_df['SessionId'].astype(str)
clicks_df = clicks_df.groupby('SessionId')['ItemId'].apply(list).reset_index(name='ItemIds')  # Group items by session
clicks_df = clicks_df[clicks_df['ItemIds'].apply(len) >= 2]  # Filter out sessions with less than 2 items
clicks_df['CategoryIds'] = clicks_df['ItemIds'].apply(
    lambda item_ids: [categories_df.loc[item_id]['CategoryName'] for item_id in item_ids])

# Create the item co-occurrence graph
graph_df = clicks_df.explode('ItemIds')
graph_df = graph_df.groupby('ItemIds')['SessionId'].apply(set).reset_index(name='Sessions')
graph_df['NumSessions'] = graph_df['Sessions'].apply(len)
graph_df = graph_df[graph_df['NumSessions'] >= 2]  # Filter out items that appear in less than 2 sessions
graph_df['Edges'] = graph_df['Sessions'].apply(
    lambda sessions: [(session1, session2) for session1 in sessions for session2 in sessions if session1 < session2])
edges = [edge for edges in graph_df['Edges'] for edge in edges]
graph = nx.Graph()
graph.add_edges_from(edges)

# Add the hierarchical category information to the graph
category_mapping = {category_name: i for i, category_name in enumerate(categories_df['CategoryName'].unique())}
graph_df['CategoryIds'] = graph_df['ItemIds'].apply(
    lambda item_ids: [category_mapping[category_name] for category_name in categories_df.loc[item_ids]['CategoryName']])
category_edges = []
for i, row in categories_df.iterrows():
    category_name = row['CategoryName']
    if not pd.isnull(category_name):
        parent_category_name = categories_df.loc[row['CategoryIdParent']]['CategoryName']
        if not pd.isnull(parent_category_name):
            category_edges.append((category_mapping[category_name], category_mapping[parent_category_name]))
graph.add_edges_from(category_edges)

# Convert the graph to a sparse adjacency matrix
adj_matrix = nx.adjacency_matrix(graph)


# Create the GCN model
def gcn_layer(inputs, adj_matrix, output_dim, activation, use_bias=True):
    # Apply the graph convolution operation
    hidden = tf.sparse.sparse_dense_matmul(adj_matrix, inputs)

    # Normalize the hidden activations
    degrees = tf.sparse.reduce_sum(adj_matrix, axis=1)
    degrees = tf.squeeze(tf.where(degrees == 0, tf.ones_like(degrees), degrees))
    hidden = hidden / tf.sqrt(degrees[:, None])

    # Apply the linear transformation and activation function
    hidden = Dense(units=output_dim, activation=activation, use_bias=use_bias)(hidden)
    return hidden


def build_model(num_items, num_categories, embedding_dim=64, hidden_dim=64):
    # Define the input layers
    item_input = Input(shape=(1,))
    category_input = Input(shape=(1,))

    # Create the item and category embeddings
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim)(item_input)
    category_embedding = Embedding(input_dim=num_categories, output_dim=embedding_dim)(category_input)

    # Combine the item and category embeddings
    inputs = tf.concat([item_embedding, category_embedding], axis=1)

    # Apply two graph convolutional layers with ReLU activations
    hidden = gcn_layer(inputs, adj_matrix, output_dim=hidden_dim, activation='relu')
    hidden = Dropout(0.5)(hidden)
    hidden = gcn_layer(hidden, adj_matrix, output_dim=hidden_dim, activation='relu')
    hidden = Dropout(0.5)(hidden)

    # Apply the final linear transformation
    output = Dense(units=1, activation='sigmoid')(hidden)

    # Define the model inputs and outputs
    inputs = [item_input, category_input]
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    optimizer = Adam(lr=0.001)
    loss = BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss)

    return model


# Convert the session data to a sequence of input and target pairs
def generate_sequence(item_ids, category_ids):
    input_sequence = []
    target_sequence = []
    for i in range(1, len(item_ids)):
        input_sequence.append((item_ids[:i], category_ids[:i]))
        target_sequence.append(int(item_ids[i] in item_ids[:i]))
    return input_sequence, target_sequence


# Split the data into train and test sets
train_size = int(len(clicks_df) * 0.8)
train_clicks_df = clicks_df[:train_size]
test_clicks_df = clicks_df[train_size:]

# Generate the train and test sequences
train_sequence = []
for _, row in train_clicks_df.iterrows():
    input_sequence, target_sequence = generate_sequence(row['ItemIds'], row['CategoryIds'])
    train_sequence += list(zip(input_sequence, target_sequence))
test_sequence = []
for _, row in test_clicks_df.iterrows():
    input_sequence, target_sequence = generate_sequence(row['ItemIds'], row['CategoryIds'])
    test_sequence += list(zip(input_sequence, target_sequence))

# Define the model hyperparameters
embedding_dim = 64
hidden_dim = 64
batch_size = 64
num_epochs = 10

# Build the model and print a summary
num_items = len(clicks_df['ItemId'].unique())
num_categories = len(categories_df)
model = build_model(num_items, num_categories, embedding_dim, hidden_dim)
print(model.summary())

# Train the model
train_inputs = np.array([inputs for inputs, _ in train_sequence])
train_targets = np.array([target for _, target in train_sequence])
test_inputs = np.array([inputs for inputs, _ in test_sequence])
test_targets = np.array([target for _, target in test_sequence])
model.fit(x=[train_inputs[:, 0], train_inputs[:, 1]], y=train_targets, batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_data=([test_inputs[:, 0], test_inputs[:, 1]], test_targets))

# Evaluate the model
test_predictions = model.predict([test_inputs[:, 0], test_inputs[:, 1]])
test_predictions = np.round(test_predictions).astype(int)
test_targets = test_targets.astype(int)

precision = precision_score(test_targets, test_predictions)
recall = recall_score(test_targets, test_predictions)
mse = mean_squared_error(test_targets, test_predictions)

print("Precision: %.4f" % precision)
print("Recall: %.4f" % recall)
print("Mean Squared Error: %.4f" % mse)
