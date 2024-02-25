import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
import gym
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix


# Data preprocessing
# Load the Yoochoose dataset and preprocess it to create session-based sequences of interactions
data = pd.read_csv('yoochoose_dataset/filtered_clicks.dat',
                   names=['session_id', 'timestamp', 'item_id', 'category'],
                   dtype={'session_id': 'int64', 'timestamp': 'str', 'item_id': 'int64', 'category': 'int64'},
                   parse_dates=['timestamp'])

# Create item and session maps
item_map = dict(zip(np.unique(data.item_id), range(len(np.unique(data.item_id)))))
session_map = dict(zip(np.unique(data.session_id), range(len(np.unique(data.session_id)))))

# Map item and session IDs
data['item_id'] = data['item_id'].map(item_map)
data['session_id'] = data['session_id'].map(session_map)

# Sort by session and timestamp
data = data.sort_values(['session_id', 'timestamp'])

# Create next item and session columns
data['next_item_id'] = data.groupby('session_id')['item_id'].shift(-1)
data['next_session_id'] = data.groupby('session_id')['session_id'].shift(-1)
data = data.dropna()

# Convert data to numpy arrays
session_ids = data['session_id'].values
item_ids = data['item_id'].values
next_item_ids = data['next_item_id'].values
next_session_ids = data['next_session_id'].values
timestamps = data['timestamp'].values

# Create a directed graph
graph = nx.DiGraph()

# Add nodes to the graph
graph.add_nodes_from(item_map.values())

# Add edges to the graph
for session_id, items in tqdm(data.groupby('session_id')['item_id']):
    items = items.values.tolist()
    for i in range(len(items)-1):
        src, dst = items[i], items[i+1]
        graph.add_edge(src, dst)
        
        
# Create dense feature matrix
num_items = len(item_map)
features = np.eye(num_items)

# Create adjacency matrix
adj_matrix = nx.adjacency_matrix(graph, nodelist=range(num_items))
adj_matrix = csr_matrix(adj_matrix)

# Get the weight values from the graph edges
edges = graph.edges(data=True)
# create edges as dictionaries with 'weight' key
edges = [(source, target, {'weight': 1.0}) for source, target in edges_list]

# modify the 'weight' value for specific edges
edges[0][2]['weight'] = 0.5 # set weight to 0.5 for the first edge

values = np.array([edge_data['weight'] for _, _, edge_data in edges], dtype=np.float32)

# Define indices and values
indices = np.transpose([adj_matrix.nonzero()[0], adj_matrix.nonzero()[1]])
indices = tf.cast(indices, 'int64') # convert indices to int64
values = tf.expand_dims(values, axis=1)
values = tf.squeeze(values)


# Set the hyperparameters
hidden_dim = 64

# Create sparse adjacency matrix
adj_sparse = tf.sparse.SparseTensor(indices=np.transpose([adj_matrix.nonzero()[0], adj_matrix.nonzero()[1]]),
                                    values=adj_matrix.data,
                                    dense_shape=adj_matrix.shape)

class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation):
        super(GraphConvolution, self).__init__()
        self.output_dim = output_dim
        self.activation = activation
        self.num_features = None  # add num_features attribute

    def build(self, input_shape):
        self.num_features = input_shape[0][-1]  # set num_features based on the last dimension of input_layer
        self.kernel = self.add_weight(
            shape=(self.num_features, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        x, indices, values = inputs
        indices = tf.cast(indices, tf.int64)
        dense_shape = (x.shape[0], self.num_features) if x.shape[0] is not None else None
        x = tf.sparse.SparseTensor(indices, values, dense_shape=dense_shape)
        x = tf.sparse.sparse_dense_matmul(x, self.kernel)
        return self.activation(x)


def build_gcn_model(num_nodes, hidden_dim):
    # Define inputs
    input_layer = Input(shape=(num_nodes,))
    indices = Input(shape=(2,), dtype=tf.int32)
    values = Input(shape=(1,), dtype=tf.float32)

    # Define layers
    gcn1 = GraphConvolution(hidden_dim, activation='relu')([input_layer, indices, values])
    dropout1 = Dropout(0.5)(gcn1)
    gcn2 = GraphConvolution(hidden_dim, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(gcn2)
    gcn3 = GraphConvolution(hidden_dim, activation='relu')(dropout2)

    # Define outputs
    output_layer = GraphConvolution(1, activation='sigmoid')(gcn3)

    # Define model
    gcn_model = Model(inputs=[input_layer, indices, values], outputs=output_layer)
    gcn_model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    return gcn_model


# Build the model
gcn_model = build_gcn_model(num_items, hidden_dim)
gcn_model.fit([features, indices_np, values], next_item_ids, epochs=num_epochs)

# Compile the model
gcn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Print the model summary
print(gcn_model.summary())


# Define the self-supervised learning task
def self_supervised_task(gcn_model, session_ids, item_ids, next_item_ids):
    # Mask the last item in each session
    mask = np.concatenate([np.where(session_ids[:-1] != session_ids[1:])[0], [len(session_ids)-1]])
    masked_item_ids = np.delete(item_ids, mask)
    masked_session_ids = np.delete(session_ids, mask)
    masked_next_item_ids = np.delete(next_item_ids, mask)

    # Predict the next item using the GCN model
    predicted_next_item_probs = gcn_model.predict([features[masked_item_ids], adj_matrix])
    predicted_next_item_probs = predicted_next_item_probs[range(len(predicted_next_item_probs)), masked_item_ids]
    predicted_next_item_probs = np.log(predicted_next_item_probs + 1e-10) # Log transform the probabilities to avoid numerical instability

    # Compute the mean squared error loss
    mse_loss = np.mean(np.square(predicted_next_item_probs - masked_next_item_ids))

    # Compute the cross-entropy loss
    ce_loss = -np.mean(np.log(predicted_next_item_probs[range(len(predicted_next_item_probs)), masked_next_item_ids] + 1e-10))

    return mse_loss, ce_loss

# Pretrain the GCN model using self-supervised learning
num_epochs = 10
batch_size = 64

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))

    # Shuffle the data
    permutation = np.random.permutation(len(session_ids))
    session_ids_shuffled = session_ids[permutation]
    item_ids_shuffled = item_ids[permutation]
    next_item_ids_shuffled = next_item_ids[permutation]

    # Iterate over batches of data
    for i in range(0, len(session_ids), batch_size):
        # Get the batch data
        batch_session_ids = session_ids_shuffled[i:i+batch_size]
        batch_item_ids = item_ids_shuffled[i:i+batch_size]
        batch_next_item_ids = next_item_ids_shuffled[i:i+batch_size]

        # Perform the self-supervised learning task
        mse_loss, ce_loss = self_supervised_task(gcn_model, batch_session_ids, batch_item_ids, batch_next_item_ids)

        # Update the GCN model weights using the optimizer
        gcn_model.train_on_batch([features[batch_item_ids], adj_matrix], batch_next_item_ids)

        # Print the loss
        print('Batch {}/{} - MSE loss: {:.4f}, CE loss: {:.4f}'.format(i//batch_size+1, np.ceil(len(session_ids)/batch_size), mse_loss, ce_loss))

        
# Define loss function
def masked_loss(y_true, y_pred):
    # Get the mask for the last item in each session
    mask = tf.not_equal(y_true, num_items)
    
    # Flatten the mask and the predictions
    mask = tf.reshape(mask, [-1])
    y_pred = tf.reshape(y_pred, [-1, num_items])
    
    # Apply the mask to the predictions and the labels
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    
    # Compute the cross-entropy loss
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked))
    
    return loss

# Define the batch size and number of epochs
batch_size = 128
epochs = 10

# Define the label encoder
label_encoder = LabelEncoder()

# Encode the item IDs
item_ids_encoded = label_encoder.fit_transform(item_ids)

# Pre-train the GCN model using self-supervised learning
for epoch in range(epochs):
    # Shuffle the data
    indices = np.random.permutation(len(session_ids))
    
    # Create a deque to store the last items in each session
    last_items = deque(maxlen=batch_size)
    
    for i in range(0, len(indices), batch_size):
        # Get the batch indices
        batch_indices = indices[i:i+batch_size]
        
        # Get the batch data
        batch_session_ids = session_ids[batch_indices]
        batch_item_ids = item_ids_encoded[batch_indices]
        batch_next_item_ids = next_item_ids[batch_indices]
        
        # Create the input features
        batch_features = features[batch_item_ids]
        batch_adj_matrix = adj_matrix[batch_item_ids][:,batch_item_ids]
        
        # Mask the last item in each session
        mask = np.ones_like(batch_next_item_ids)
        for j, last_item in enumerate(last_items):
            if last_item is not None:
                mask[j] = 0
                batch_next_item_ids[j] = last_item
        
        # Update the last items deque
        last_items.extend(batch_next_item_ids)
        
        # Train the GCN model
        gcn_model.train_on_batch([batch_features, batch_adj_matrix], batch_next_item_ids, sample_weight=mask)
        
    

# Split the data into training and testing sets
train_session_ids, test_session_ids, train_item_ids, test_item_ids, train_next_item_ids, test_next_item_ids = train_test_split(session_ids, item_ids, next_item_ids, test_size=0.2)

# Train the GCN model on the training set
gcn_model.fit([features, adj_matrix], train_item_ids, epochs=10, batch_size=128)

# Evaluate the performance of the model on the testing set
test_pred = np.argmax(gcn_model.predict([features, adj_matrix]), axis=1)
test_acc = accuracy_score(test_item_ids, test_pred)
test_prec = precision_score(test_item_ids, test_pred, average='macro')
test_rec = recall_score(test_item_ids, test_pred, average='macro')
test_f1 = f1_score(test_item_ids, test_pred, average='macro')
print('Test accuracy:', test_acc)
print('Test precision:', test_prec)
print('Test recall:', test_rec)
print('Test F1-score:', test_f1)

