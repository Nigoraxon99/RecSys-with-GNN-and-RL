# First, make sure you have all the required dependencies installed,
# including TensorFlow, Pandas, Numpy, Scipy, and NetworkX


import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque

# Load clickstream file
data = pd.read_csv('yoochoose_dataset/yoochoose-clicks.dat',
                   names=['session_id', 'timestamp', 'item_id', 'category'],
                   dtype={'session_id': 'int64', 'timestamp': 'str', 'item_id': 'int64', 'category': 'int64'},
                   parse_dates=['timestamp'])

# # Remove sessions with less than 2 clicks
# session_lengths = data.groupby('session_id').size()
# data = data[np.in1d(data.session_id, session_lengths[session_lengths >= 2].index)]

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

# Create graph
graph = nx.Graph()

# Add edges between items that co-occur in the same session
for session_id in np.unique(session_ids):
    items_in_session = item_ids[session_ids == session_id]
    for i in range(len(items_in_session)):
        for j in range(i + 1, len(items_in_session)):
            if not graph.has_edge(items_in_session[i], items_in_session[j]):
                graph.add_edge(items_in_session[i], items_in_session[j], weight=0)
            graph[items_in_session[i]][items_in_session[j]]['weight'] += 1

# Normalize edge weights
for u, v, d in graph.edges(data=True):
    d['weight'] /= np.sqrt(graph.degree(u) * graph.degree(v))


# Define GNN model

class GNN(tf.keras.Model):
    def __init__(self, num_nodes, embedding_dim, num_layers):
        super(GNN, self).__init__()
        self.embedding = layers.Embedding(num_nodes, embedding_dim)
        self.layers = [layers.Dense(embedding_dim, activation='relu') for _ in range(num_layers)]

    def call(self, inputs, **kwargs):
        node_ids, adj_matrix = inputs
        x = self.embedding(node_ids)
        for layer in self.layers:
            x = layer(tf.sparse.sparse_dense_matmul(adj_matrix, x))
        return x


# Define contrastive loss function
def contrastive_loss(y_true, y_pred, temperature):
    logits = tf.matmul(y_pred, tf.transpose(y_pred)) / temperature
    labels = tf.one_hot(tf.range(tf.shape(y_pred)[0]), tf.shape(y_pred)[0] * 2)
    mask = 1 - tf.eye(tf.shape(y_pred)[0], dtype=tf.int32)
    labels = tf.reshape(labels, (-1, tf.shape(y_pred)[0] * 2))
    mask = tf.reshape(mask, (-1,))
    labels = tf.boolean_mask(labels, mask)
    logits = tf.boolean_mask(logits, mask)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    return tf.reduce_mean(loss)


# Define hyperparameters
num_nodes = adj_matrix.shape[0]
embedding_dim = 32
num_layers = 2
temperature = 0.07
learning_rate = 0.001
num_epochs = 10
batch_size = 128

# Create adjacency matrix
adj_matrix = nx.to_scipy_sparse_matrix(graph, weight='weight', dtype=np.float32)
adj_matrix = tf.sparse.SparseTensor(indices=np.array([adj_matrix.row, adj_matrix.col]).T,
                                    values=adj_matrix.data,
                                    dense_shape=adj_matrix.shape)

# Create GNN model
gnn = GNN(num_nodes, embedding_dim, num_layers)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Train GNN model
for epoch in range(num_epochs):
    for i in range(0, len(session_ids), batch_size):
        batch_node_ids = session_ids[i:i + batch_size]
        batch_adj_matrix = tf.sparse.SparseTensor(indices=adj_matrix.indices,
                                                  values=adj_matrix.values[i:i + batch_size],
                                                  dense_shape=adj_matrix.dense_shape)
        batch_features = gnn([batch_node_ids, batch_adj_matrix])
        batch_labels = tf.concat([batch_features, batch_features], axis=0)
        batch_loss = contrastive_loss(None, batch_labels, temperature)
        grads = tf.gradients(batch_loss, gnn.trainable_weights)
        optimizer.apply_gradients(zip(grads, gnn.trainable_weights))


# Define session-based recommender system with reinforcement learning
class RecommenderSystem:
    def __init__(self, gnn, item_map, gamma=0.9, alpha=0.1):
        self.gnn = gnn
        self.item_map = item_map
        self.gamma = gamma
        self.alpha = alpha
        self.replay_buffer = deque(maxlen=10000)
        self.session_history = []

    def recommend(self, session_items):
        session_node_ids = [self.item_map[item] for item in session_items if item in self.item_map]
        if len(session_node_ids) == 0:
            return []
        session_adj_matrix = tf.sparse.SparseTensor(indices=adj_matrix.indices,
                                                    values=adj_matrix.values[np.isin(adj_matrix.row, session_node_ids)],
                                                    dense_shape=adj_matrix.dense_shape)
        session_features = gnn([session_node_ids, session_adj_matrix])
        item_scores = np.matmul(session_features, gnn.embedding.weights[0].numpy().T)
        item_scores[np.isin(np.arange(len(item_map)), session_node_ids)] = -np.inf
        item_indices = np.argsort(item_scores)[::-1]
        return [item_map[i] for i in item_indices[:10]]

    def update(self, session_items, reward):
        session_node_ids = [self.item_map[item] for item in session_items if item in self.item_map]
        if len(session_node_ids) == 0:
            return
        session_adj_matrix = tf.sparse.SparseTensor(indices=adj_matrix.indices,
                                                    values=adj_matrix.values[np.isin(adj_matrix.row, session_node_ids)],
                                                    dense_shape=adj_matrix.dense_shape)
        session_features = gnn([session_node_ids, session_adj_matrix])
        item_scores = np.matmul(session_features, gnn.embedding.weights[0].numpy().T)
        item_indices = np.argsort(item_scores)[::-1]
        item_probs = np.exp(item_scores) / np.sum(np.exp(item_scores))
        item_probs[np.isin(np.arange(len(item_map)), session_node_ids)] = 0
        item_probs = item_probs / np.sum(item_probs)
        item_rewards = np.zeros(len(item_map))
        item_rewards[item_indices[:10]] = reward
        self.replay_buffer.append((session_features.numpy(), item_probs, item_rewards))
        self.session_history.append(session_node_ids)

        if len(self.replay_buffer) == self.replay_buffer.maxlen:
            for i in range(self.replay_buffer.maxlen):
                session_features, item_probs, item_rewards = self.replay_buffer[i]
                discounted_rewards = np.zeros(len(item_map))
                running_reward = 0
                for j in range(len(item_map)):
                    if item_rewards[j] != 0:
                        running_reward = item_rewards[j]
                    else:
                        running_reward = running_reward * self.gamma
                    discounted_rewards[j] = running_reward
                item_values = np.sum(np.exp(np.matmul(session_features, gnn.embedding.weights[0].numpy().T)) * discounted_rewards,axis=1)
                item_grads = tf.gradients(tf.math.log(item_probs), gnn.trainable_weights, grad_ys=item_values)
                optimizer.apply_gradients(zip(item_grads, gnn.trainable_weights))

            self.replay_buffer.clear()
            self.session_history.clear()


# Create session-based recommender system
recommender = RecommenderSystem(gnn, item_map)

# Define hyperparameters for reinforcement learning
num_sessions = 100
max_session_length = 10
optimizer = tf.optimizers.Adam(learning_rate=0.001)
mean_squared_error = tf.keras.losses.MeanSquaredError()


# Train GNN with reinforcement learning
def compute_reward(recommended_items, purchased_items):
    """
    Computes the reward based on the precision and recall of the recommended items.
    """
    if len(recommended_items) == 0:
        return 0.0

    precision = len(set(recommended_items).intersection(set(purchased_items))) / float(len(recommended_items))
    recall = len(set(recommended_items).intersection(set(purchased_items))) / float(len(purchased_items))

    return precision * recall


for i in range(num_sessions):
    session_items = np.random.choice(item_map.keys(), size=max_session_length, replace=False)
    recommended_items = recommender.recommend(session_items)
    reward = compute_reward(session_items, recommended_items)
    recommender.update(session_items, reward)
    if i % 10 == 0:
        mse = 0
        for session_node_ids in recommender.session_history:
            session_adj_matrix = tf.sparse.SparseTensor(indices=adj_matrix.indices,values=adj_matrix.values[np.isin(adj_matrix.row, session_node_ids)],
                                                        dense_shape=adj_matrix.dense_shape)
            session_features = gnn([session_node_ids, session_adj_matrix])
            item_scores = np.matmul(session_features, gnn.embedding.weights[0].numpy().T)
            item_indices = np.argsort(item_scores)[::-1]
            item_probs = np.exp(item_scores) / np.sum(np.exp(item_scores))
            item_probs[np.isin(np.arange(len(item_map)), session_node_ids)] = 0
            item_probs = item_probs / np.sum(item_probs)
            item_rewards = np.zeros(len(item_map))
            item_rewards[item_indices[:10]] = 1
            mse += mean_squared_error(item_probs, item_rewards).numpy()
        print('MSE:', mse / len(recommender.session_history))
        recommender.session_history.clear()

# Evaluate the performance of the GNN using precision, recall, and mean squared error
buy_events = pd.read_csv('yoochoose_dataset/yoochoose-buys.dat', header=None, usecols=[0, 2, 3], names=['SessionId', 'ItemId', 'Price'])
buy_events = buy_events[buy_events['ItemId'].isin(item_map.keys())].groupby('SessionId')['ItemId'].apply(set).reset_index()
buy_events['ItemId'] = buy_events['ItemId'].apply(lambda x: list(x))

test_clickstream = data.sample(frac=0.2)
test_clickstream['ItemId'] = test_clickstream['ItemId'].apply(lambda x: item_map.get(x, -1))
test_clickstream = test_clickstream[test_clickstream['ItemId'] != -1]

test_sessions = test_clickstream.groupby('SessionId')['ItemId'].apply(list).reset_index()

test_session_graphs = []
for session_items in test_sessions['ItemId']:
    session_nodes = set(session_items)
    session_adj_matrix = adj_matrix.copy()
    session_adj_matrix = tf.sparse.SparseTensor(indices=session_adj_matrix.indices,
                                                values=session_adj_matrix.values[
                                                    np.isin(session_adj_matrix.row, session_nodes)],
                                                dense_shape=session_adj_matrix.dense_shape)
    session_features = gnn([session_nodes, session_adj_matrix])
    test_session_graphs.append(session_features.numpy())

precision = 0
recall = 0
mse = 0
item_map_inv = {v: k for k, v in item_map.items()}
for i, session_graph in enumerate(test_session_graphs):
    session_items = test_sessions['ItemId'][i]
    item_scores = np.matmul(session_graph, gnn.embedding.weights[0].numpy().T)
    item_indices = np.argsort(item_scores)[::-1]
    recommended_items = item_indices[:10]
    recommended_items = [item_map_inv[x] for x in recommended_items if x in item_map_inv]
    purchased_items = buy_events[buy_events['SessionId'] == test_sessions['SessionId'][i]]['ItemId'].iloc[0]
    true_positives = set(recommended_items).intersection(purchased_items)
    precision += len(true_positives) / len(recommended_items)
    recall += len(true_positives) / len(purchased_items)
    mse += mean_squared_error(item_scores, [1 if x in purchased_items else 0 for x in range(len(item_map))]).numpy()
precision /= len(test_session_graphs)
recall /= len(test_session_graphs)
mse /= len(test_session_graphs)
print('Precision:', precision)
print('Recall:', recall)
print('MSE:', mse)
