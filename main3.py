import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision
import numpy as np

df = pd.read_csv('dataset.csv')

# Convert user_id, item_id, and session_id to numerical indices
df['user_id'] = pd.Categorical(df['user_id'])
df['item_id'] = pd.Categorical(df['item_id'])
df['session_id'] = pd.Categorical(df['session_id'])

df['user_idx'] = df['user_id'].cat.codes
df['item_idx'] = df['item_id'].cat.codes
df['session_idx'] = df['session_id'].cat.codes

# Create input and output tensors
input_tensor = np.column_stack((df['user_idx'], df['item_idx'], df['session_idx']))
output_tensor = np.array(df['eventdate'])

# Define the model architecture
embedding_size = 32
num_hidden_units = 64

user_input = tf.keras.layers.Input(shape=(1,))
user_embedding = tf.keras.layers.Embedding(input_dim=len(df['user_idx'].unique()), output_dim=embedding_size)(
    user_input)

item_input = tf.keras.layers.Input(shape=(1,))
item_embedding = tf.keras.layers.Embedding(input_dim=len(df['item_idx'].unique()), output_dim=embedding_size)(
    item_input)

session_input = tf.keras.layers.Input(shape=(1,))
session_embedding = tf.keras.layers.Embedding(input_dim=len(df['session_idx'].unique()), output_dim=embedding_size)(
    session_input)

concatenated = tf.keras.layers.concatenate([user_embedding, item_embedding, session_embedding])
flatten = tf.keras.layers.Flatten()(concatenated)

hidden = tf.keras.layers.Dense(num_hidden_units, activation='relu')(flatten)
output = tf.keras.layers.Dense(1, activation='linear')(hidden)

# Create the model
model = tf.keras.Model(inputs=[user_input, item_input, session_input], outputs=output)

# Define evaluation metrics
metrics = ['mae', Recall(name='recall'), Precision(name='precision')]

# Compile the model with the defined metrics
model.compile(loss='mse', optimizer='adam', metrics=metrics)

# Train the model with the defined metrics
batch_size = 64
num_epochs = 100

model.fit(x=[input_tensor[:, 0], input_tensor[:, 1], input_tensor[:, 2]], y=output_tensor, batch_size=batch_size,
          epochs=num_epochs)

# Split the dataset into training and test sets
test_size = 0.2
train_data = df.sample(frac=1 - test_size, random_state=42)
test_data = df.drop(train_data.index)

# Create input and output tensors for test data
test_input_tensor = test_data[['user_int', 'item_int', 'timeframe']].to_numpy()
test_output_tensor = test_data['eventdate'].to_numpy()

# Evaluate the model on test data
metrics_values = model.evaluate(x=[test_input_tensor[:, 0], test_input_tensor[:, 1], test_input_tensor[:, 2]],
                                y=test_output_tensor, batch_size=batch_size)
print(f"Test MAE: {metrics_values[0]}")
print(f"Test recall: {metrics_values[1]}")
print(f"Test precision: {metrics_values[2]}")
