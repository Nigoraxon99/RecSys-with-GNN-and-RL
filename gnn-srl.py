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
import tensorflow.keras.backend as K
from sklearn.preprocessing import LabelEncoder
import gym
from gym import spaces
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from scipy.sparse import coo_matrix
import scipy as sp
import copy
import csv
from collections import Counter


# Define the file paths
data_path = 'yoochoose_dataset/yoochoose-clicks.dat'
output_path = 'yoochoose_dataset/filtered_clicks.dat'

# Open the input and output files
with open(data_path, 'r') as f_in, open(output_path, 'w', newline='') as f_out:
    reader = csv.reader(f_in, delimiter=',')
    writer = csv.writer(f_out, delimiter=',')
    
    session_dict = {}
    item_counts = Counter()
    
    # Loop through the rows in the input file
    for row in reader:
        # Extract the session_id and item_id
        session_id = row[0]
        item_id = row[2]
        
        # Update the item count
        item_counts[item_id] += 1
        
        # Check if the session_id already exists in the dictionary
        if session_id in session_dict:
            # If it exists, append the item_id to the existing list
            session_dict[session_id].append(item_id)
        else:
            # If it doesn't exist, create a new list with the current item_id
            session_dict[session_id] = [item_id]
        
        # Check if the session length is at least 2 and all items in the session appear at least 5 times in the dataset
        if len(session_dict[session_id]) >= 2 and all(item_counts[item] >= 5 for item in session_dict[session_id]):
            # If it is, write the row to the output file
            writer.writerow(row)



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
session_ids = data['session_id'].values.astype('int32')
item_ids = data['item_id'].values.astype('int32')
next_item_ids = data['next_item_id'].values.astype('int32')
next_session_ids = data['next_session_id'].values.astype('int32')
timestamps = data['timestamp'].values

# Create a directed graph
graph = nx.DiGraph()

# Add nodes to the graph
graph.add_nodes_from(item_map.values())

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
    

num_items = len(item_map)

    
embedding_dim = 32
num_layers = 2
hidden_dim = 32


class GNNActor(tf.keras.Model):
    def __init__(self, num_items, num_features, num_edge_features, hidden_dim):
        super(GNNActor, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(None, num_features + num_edge_features), sparse=True)
        self.num_items = num_items
        self.num_features = num_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim

        # Define node embedding layer
        self.node_embedding = tf.keras.layers.Embedding(input_dim=num_items, output_dim=hidden_dim, input_length=1)

        # Define graph convolutional layers
        self.gcn_layer1 = tf.keras.layers.Dense(units=hidden_dim, activation='relu')
        self.gcn_layer2 = tf.keras.layers.Dense(units=hidden_dim, activation='relu')
        
        # Define final prediction layer
        self.prediction_layer = tf.keras.layers.Dense(units=num_items, activation='softmax')

    def call(self, inputs):
        # Check input shape and extract node and edge features accordingly
        if len(inputs.shape) == 1:
            node_features = tf.expand_dims(tf.range(self.num_items), axis=-1)
            edge_features = inputs
        else:
            node_features, edge_features = tf.unstack(inputs, axis=1)
            edge_features = tf.transpose(edge_features, perm=[0, 2, 1]) # (batch_size, num_edge_features, num_edges)

        # Node embedding layer
        node_embeddings = self.node_embedding(node_features) # (batch_size, 1, hidden_dim)

        # Graph convolutional layers
        hidden1 = self.gcn_layer1(tf.linalg.matmul(edge_features, node_embeddings)) # (batch_size, num_edge_features, hidden_dim)
        hidden2 = self.gcn_layer2(tf.linalg.matmul(edge_features, hidden1)) # (batch_size, num_edge_features, hidden_dim)

        # Concatenate node features and hidden layers
        concat_features = tf.concat([node_embeddings, hidden1, hidden2], axis=-1) # (batch_size, 1, 3*hidden_dim)

        # Final prediction layer
        predictions = self.prediction_layer(tf.squeeze(concat_features, axis=1)) # (batch_size, num_items)

        return predictions



class DQNCritic(tf.keras.Model):
    def __init__(self, num_items, hidden_dim):
        super(DQNCritic, self).__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        
        # Define dense layers
        self.dense1 = tf.keras.layers.Dense(units=hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, inputs):
        # Pass input through dense layers
        x = self.dense1(inputs)
        x = self.dense2(x)
        
        # Reshape output to (batch_size, num_items)
        q_values = tf.reshape(x, shape=(-1, self.num_items))
        
        return q_values
    
class RecommenderEnv(gym.Env):
    def __init__(self, data, item_map, session_map):
        self.data = data
        self.item_map = item_map
        self.session_map = session_map
        self.num_items = len(item_map)
        self.action_space = spaces.Discrete(self.num_items)
        self.observation_space = spaces.MultiDiscrete([self.num_items] * 10) # assumes max session length of 10
        self.reset()

    def reset(self):
        self.session = []
        self.current_item = None
        self.current_reward = 0
        self.current_step = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        item_id = self.item_map[action]
        self.session.append(item_id)
        if self.current_item is not None:
            self.current_reward = int(item_id == self.current_item['next_item_id'])
        self.current_item = self.data[self.data['item_id'] == item_id].iloc[0]
        self.current_step += 1
        if self.current_step == len(self.current_item['next_session_id']):
            self.done = True
        return self._get_state(), self.current_reward, self.done, {}

    def _get_state(self):
        state = np.zeros((10,))
        for i, item_id in enumerate(self.session[-10:]):
            state[i] = item_id
        return state
    
# Define instance of environment 
env = RecommenderEnv(data, item_map, session_map)
    
# Define the GNNActor model
gnn_actor = GNNActor(num_items, num_features=1, num_edge_features=1, hidden_dim=32)

# Define the DQNCritic model
dqn_critic = DQNCritic(num_items, hidden_dim)

# Define the loss function for the DQN
def dqn_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Define the hyperparameters
learning_rate = 0.001
batch_size = 32
discount_factor = 0.99
exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
num_episodes = 1000
replay_buffer_size = 100000

# Define the optimizer for both models
optimizer = Adam(learning_rate)

# Compile both models
gnn_actor.compile(optimizer=optimizer, loss='categorical_crossentropy')
dqn_critic.compile(optimizer=optimizer, loss=dqn_loss)

# Define a function to update the DQN model weights
def update_dqn_model(current_state, action, reward, next_state, done):
    # Compute the target Q-value for the current state and action
    q_target = reward + gamma * tf.reduce_max(dqn_critic.predict([next_state, gnn_actor(next_state)]), axis=1) * (1 - done)

    # Update the DQN model weights using the Bellman equation
    y = dqn_critic.predict([current_state, gnn_actor(current_state)])
    y[np.arange(len(y)), action] = q_target
    dqn_critic.fit([current_state, gnn_actor(current_state)], y, verbose=0)

# Define the replay buffer
replay_buffer = deque(maxlen=replay_buffer_size)

# Define a function to sample a batch of experiences from the replay buffer
def sample_batch(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    current_states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(current_states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# Train the GNNActor and DQNCritic models jointly for session-based recommendation

# Define the function to sample a batch of experiences
def sample_batch(replay_buffer, batch_size):
    batch = random.sample(replay_buffer, batch_size)
    states = np.array([experience[0] for experience in batch])
    actions = np.array([experience[1] for experience in batch])
    rewards = np.array([experience[2] for experience in batch])
    next_states = np.array([experience[3] for experience in batch])
    dones = np.array([experience[4] for experience in batch])
    return states, actions, rewards, next_states, dones

# Define the function to compute the target Q-values
@tf.function
def compute_target_q_values(next_states, next_gnn_embeddings):
    next_actions = actor_model([next_gnn_embeddings, next_states])
    next_q_values = critic_model([next_states, next_actions])
    return rewards + discount_factor * (1 - dones) * tf.squeeze(next_q_values)

def compute_gradients(batch):
    # Unpack the batch of experiences
    states, actions, rewards, next_states, dones = batch

    # Compute the logits for the GNNActor
    node_features = tf.convert_to_tensor(states)
    edge_features = tf.convert_to_tensor(np.array([graph.edges[u, v]['weight'] for u, v in graph.edges]))
    gnn_actor = GNNActor(num_items, num_features, hidden_dim)
    with tf.GradientTape() as tape:
        logits = gnn_actor([node_features, edge_features])
        actor_loss = tf.keras.losses.sparse_categorical_crossentropy(actions, logits)

    # Compute gradients for the GNNActor
    actor_gradients = tape.gradient(actor_loss, gnn_actor.trainable_variables)

    # Compute the Q-values for the DQNCritic
    state_features = np.concatenate((states, np.eye(num_items)[actions]), axis=-1)
    next_state_features = np.concatenate((next_states, np.zeros((len(next_states), num_items))), axis=-1)
    q_critic = DQNCritic(num_items, num_features + 1, hidden_dim)
    with tf.GradientTape() as tape:
        q_values = q_critic(state_features)
        next_q_values = q_critic(next_state_features)
        target_q_values = rewards + (1 - dones) * discount_factor * tf.reduce_max(next_q_values, axis=-1)
        critic_loss = tf.keras.losses.mean_squared_error(target_q_values, q_values)

    # Compute gradients for the DQNCritic
    critic_gradients = tape.gradient(critic_loss, q_critic.trainable_variables)

    return actor_gradients, critic_gradients
def apply_gradients(actor_gradients, critic_gradients):
    # Apply gradients to the GNNActor
    gnn_actor_optimizer.apply_gradients(zip(actor_gradients, gnn_actor.trainable_variables))

    # Apply gradients to the DQNCritic
    q_critic_optimizer.apply_gradients(zip(critic_gradients, q_critic.trainable_variables))


def update_target_networks(gnn_actor, dqn_critic, target_gnn_actor, target_dqn_critic, tau):
    for target_param, param in zip(target_gnn_actor.parameters(), gnn_actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for target_param, param in zip(target_dqn_critic.parameters(), dqn_critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    
# Set hyperparameters
batch_size = 32
gamma = 0.99  # discount factor
tau = 0.001  # target network update rate
actor_lr = 0.001
critic_lr = 0.001
max_episodes = 1000  # maximum number of episodes to run
max_steps_per_episode = 100  # maximum number of steps per episode
replay_buffer_size = int(1e6)

# Create replay buffer
replay_buffer = deque(maxlen=replay_buffer_size)

# Create GNNActor and DQNCritic models
gnn_actor = GNNActor(num_items, num_features=1, num_edge_features=1, hidden_dim=32)
dqn_critic = DQNCritic(num_items, hidden_dim)

# Define optimizers
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

# Initialize target networks
# target_gnn_actor = tf.keras.models.clone_model(gnn_actor)
# target_dqn_critic = tf.keras.models.clone_model(dqn_critic)
# target_gnn_actor.set_weights(gnn_actor.get_weights())
# target_dqn_critic.set_weights(dqn_critic.get_weights())

target_gnn_actor = GNNActor(num_items, num_features=1, num_edge_features=1, hidden_dim=32)
target_gnn_actor.set_weights(gnn_actor.get_weights())
target_dqn_critic = DQNCritic(num_items, hidden_dim)
target_dqn_critic.set_weights(dqn_critic.get_weights())


# Define training loop
for episode in tqdm(range(max_episodes)):
    # Reset environment and get initial state
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps_per_episode):
        # Sample action from GNNActor
        state_sparse = tf.sparse.from_dense(state)
        action_probs = gnn_actor(state_sparse)
        action = tf.squeeze(action, axis=1)
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action.numpy())
        
        # Update episode reward
        episode_reward += reward
        
        # Add experience to replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        
        # Sample batch from replay buffer
        if len(replay_buffer) >= batch_size:
            # Sample batch of experiences from replay buffer
            batch = np.array(random.sample(replay_buffer, batch_size), dtype=object)
            
            # Compute gradients for GNNActor and DQNCritic
            actor_gradients, critic_gradients = compute_gradients(batch)
            
            # Apply gradients to GNNActor and DQNCritic
            apply_gradients(actor_gradients, critic_gradients, actor_optimizer, critic_optimizer)
            
            # Update target networks
            update_target_networks(gnn_actor, dqn_critic, target_gnn_actor, target_dqn_critic, tau)
        
        # Update state
        state = next_state
        
        # Check if episode is done
        if done:
            break
    
    # Print episode reward
    print(f'Episode {episode}: {episode_reward}')
    
