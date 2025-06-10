from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import Sequence

# Configuration
csv_path = "/Users/isaacigbokwe/Documents/halo/halo/data/processed_data/train_labels.csv"
data_dir = "/Users/isaacigbokwe/Documents/halo/halo/data/processed_data"
batch_size = 2  # Lowered for 8GB RAM
num_frames = 20
input_shape = (224, 224, 3)

# Load label CSV
df = pd.read_csv(csv_path)

# Define custom data generator
class ClipGenerator(Sequence):
    def __init__(self, df, batch_size):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch_df.iterrows():
            clip_path = os.path.join(data_dir, row['filename'])
            clip = np.load(clip_path)
            X.append(clip)
            y.append(row['label'])
        return np.array(X), np.array(y)

# Model architecture
cnn = ResNet50(include_top=False, pooling='avg', input_shape=input_shape)
cnn.trainable = False

model = models.Sequential([
    layers.TimeDistributed(cnn, input_shape=(num_frames,) + input_shape),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

generator = ClipGenerator(df, batch_size=batch_size)
from tensorflow.keras.callbacks import CSVLogger
csv_logger = CSVLogger('/Users/isaacigbokwe/Documents/halo/halo/models/training_log.csv', append=True)
# Split into training and validation
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_generator = ClipGenerator(train_df, batch_size=batch_size)
val_generator = ClipGenerator(val_df, batch_size=batch_size)

# Train model with generators
model.fit(train_generator, validation_data=val_generator, epochs=5, callbacks=[csv_logger])
          
model.save("/Users/isaacigbokwe/Documents/halo/halo/models/halo_model.h5")