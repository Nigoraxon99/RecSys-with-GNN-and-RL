import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('data/retailrocket/rsc15_train_full.txt', sep='\t', header=None,
                 names=['sessionid', 'timestamp', 'itemid', 'category'])

# Filter out sessions with only one item
session_lengths = df.groupby('sessionid').size()
df = df[np.in1d(df['sessionid'], session_lengths[session_lengths > 1].index)]

# Encode the categorical variables
encoder = LabelEncoder()
df['sessionid'] = encoder.fit_transform(df['sessionid'])
df['itemid'] = encoder.fit_transform(df['itemid'])

# Split the data into training and validation sets
valid_fraction = 0.1
valid_mask = np.random.choice([True, False], size=len(df), p=[valid_fraction, 1 - valid_fraction])
valid_df = df[valid_mask]
train_df = df[~valid_mask]

# Define the GNN based model #####################
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the input layer
inputs = Input(shape=(None,), dtype=tf.int32, name='input')

# Define the embedding layer
emb_dim = 32
embedding = Embedding(input_dim=num_items, output_dim=emb_dim, name='embedding')(inputs)

# Define the GNN layers
units_per_layer = 64
dropout_rate = 0.2
gnn_layer1 = tf.keras.layers.GRU(units_per_layer, dropout=dropout_rate, return_sequences=True)(embedding)
gnn_layer2 = tf.keras.layers.GRU(units_per_layer, dropout=dropout_rate)(gnn_layer1)

# Define the output layer
output = Dense(num_items, activation='softmax', name='output')(gnn_layer2)

# Define the model
model = Model(inputs=inputs, outputs=output)

# Print the model summary
model.summary()


# Define the self-supervised loss function ################

def masked_crossentropy(y_true, y_pred):
    # Mask the padded items
    mask = tf.math.not_equal(y_true, 0)

    # Flatten the inputs
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, num_items])

    # Mask the inputs
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    # Compute the cross-entropy loss
    crossentropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)

    # Compute the masked loss
    masked_loss = tf.reduce_sum(crossentropy * mask) / tf.reduce_sum(tf.cast(mask, tf.float32))

    return masked_loss


# Define the reinforcement learning loss function


def rl_loss(y_true, y_pred):
    # Compute the discounted sum of rewards
    gamma = 0.99
    discounted_rewards = tf.cumsum(y_true[:, ::-1] * gamma ** tf.range(tf.shape(y_true)[1], dtype=tf.float32), axis=1)[
                         :, ::-1]

    # Compute the mean squared error between the predicted and target values
    mse = tf.keras.losses.mean_squared_error(discounted_rewards, y_pred)
    return mse


# Compile the model and train it using self-supervised and reinforcement learning

# Compile the model

model.compile(optimizer=Adam(lr=0.001), loss={'output': masked_crossentropy, 'rl_output': rl_loss})

# Train the model

batch_size = 64
num_epochs = 10
for epoch in range(num_epochs):
    for batch in range(len(train_df) // batch_size):
        # Get the batch data
        batch_df = train_df.iloc[batch * batch_size:(batch + 1) * batch_size]
        # Get the input and target sequences
        input_sequences = batch_df.groupby('sessionid')['itemid'].apply(list).values
        target_sequences = np.zeros((len(input_sequences), max_sequence_length), dtype=np.int32)
        for i, seq in enumerate(input_sequences):
            target_sequences[i, :len(seq)] = seq

        # Compute the target rewards
        target_rewards = np.zeros((len(input_sequences), max_sequence_length), dtype=np.float32)
        for i, seq in enumerate(input_sequences):
            for j in range(len(seq)):
                target_rewards[i, j] = compute_reward(seq[:j], seq[j:])

        # Train the model on the batch
        loss = model.train_on_batch(input_sequences, {'output': target_sequences, 'rl_output': target_rewards})

    # Evaluate the model on the validation set
    input_sequences = valid_df.groupby('sessionid')['itemid'].apply(list).values
    target_sequences = np.zeros((len(input_sequences), max_sequence_length), dtype=np.int32)
    for i, seq in enumerate(input_sequences):
        target_sequences[i, :len(seq)] = seq
    metrics = model.evaluate(input_sequences,
                             {'output': target_sequences, 'rl_output': np.zeros_like(target_sequences)}, verbose=0)
    print(f'Epoch {epoch + 1}/{num_epochs}: loss = {loss:.4f}, val_loss = {metrics:.4f}')

# Evaluate the model on the test set
input_sequences = test_df.groupby('sessionid')['itemid'].apply(list).values
target_sequences = np.zeros((len(input_sequences), max_sequence_length), dtype=np.int32)
for i, seq in enumerate(input_sequences):
    target_sequences[i, :len(seq)] = seq
y_pred = model.predict(input_sequences)['output']
precision = precision_at_k(y_true=target_sequences, y_pred=y_pred, k=10)
recall = recall_at_k(y_true=target_sequences, y_pred=y_pred, k=10)
mse = mean_squared_error(target_sequences, y_pred)

print(f'Precision@10 = {precision:.4f}, Recall@10 = {recall:.4f}, MSE = {mse:.4f}')

