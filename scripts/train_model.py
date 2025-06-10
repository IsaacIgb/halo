import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
csv_path = "/Users/isaacigbokwe/Documents/halo/halo/data/processed_data/train_labels.csv"
data_dir = "/Users/isaacigbokwe/Documents/halo/halo/data/processed_data"
model_dir = "/Users/isaacigbokwe/Documents/halo/halo/models"
os.makedirs(model_dir, exist_ok=True)

batch_size = 2  # Lowered for 8GB RAM
num_frames = 5   # Updated for EfficientNet preprocessing
input_shape = (224, 224, 3)

# Load label CSV
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} samples")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Create binary labels (suspicious vs normal) while keeping multi-class for learning
df['binary_label'] = (df['label'] > 0).astype(int)  # 0=normal, 1=suspicious
print(f"\nBinary distribution:\n{df['binary_label'].value_counts()}")

# Define custom data generator
class ClipGenerator(Sequence):
    def __init__(self, df, batch_size, multi_class=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.multi_class = multi_class
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        
        for _, row in batch_df.iterrows():
            clip_path = os.path.join(data_dir, row['filename'])
            try:
                clip = np.load(clip_path)
                # Apply ImageNet normalization for EfficientNet
                clip = tf.keras.applications.efficientnet.preprocess_input(clip * 255.0)
                X.append(clip)
                
                if self.multi_class:
                    # Multi-class: 8 classes (0=normal, 1-7=suspicious types)
                    y.append(row['label'])
                else:
                    # Binary: 0=normal, 1=suspicious
                    y.append(row['binary_label'])
                    
            except Exception as e:
                print(f"Error loading {clip_path}: {str(e)}")
                continue
                
        return np.array(X), np.array(y)

# Multi-class model architecture (8 classes)
print("\nBuilding multi-class model...")
efficientnet = EfficientNetB0(
    include_top=False, 
    pooling='avg', 
    input_shape=input_shape,
    weights='imagenet'
)
efficientnet.trainable = False  # Freeze pretrained weights

multi_class_model = models.Sequential([
    layers.TimeDistributed(efficientnet, input_shape=(num_frames,) + input_shape),
    layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(8, activation='softmax')  # 8 classes
])

multi_class_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Binary model architecture (suspicious vs normal)
print("Building binary model...")
binary_model = models.Sequential([
    layers.TimeDistributed(efficientnet, input_shape=(num_frames,) + input_shape),
    layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

binary_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"\nTrain samples: {len(train_df)}, Validation samples: {len(val_df)}")

# Data generators
train_generator_multi = ClipGenerator(train_df, batch_size=batch_size, multi_class=True)
val_generator_multi = ClipGenerator(val_df, batch_size=batch_size, multi_class=True)

train_generator_binary = ClipGenerator(train_df, batch_size=batch_size, multi_class=False)
val_generator_binary = ClipGenerator(val_df, batch_size=batch_size, multi_class=False)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-7
)

# Train multi-class model
print("\n" + "="*50)
print("TRAINING MULTI-CLASS MODEL (8 classes)")
print("="*50)

multi_csv_logger = CSVLogger(os.path.join(model_dir, 'multi_class_training_log.csv'))

multi_history = multi_class_model.fit(
    train_generator_multi,
    validation_data=val_generator_multi,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler, multi_csv_logger],
    verbose=1
)

# Save multi-class model
multi_class_model.save(os.path.join(model_dir, "halo_multiclass_model.h5"))
print("Multi-class model saved!")

# Train binary model
print("\n" + "="*50)
print("TRAINING BINARY MODEL (suspicious vs normal)")  
print("="*50)

binary_csv_logger = CSVLogger(os.path.join(model_dir, 'binary_training_log.csv'))

binary_history = binary_model.fit(
    train_generator_binary,
    validation_data=val_generator_binary,
    epochs=50,
    callbacks=[early_stopping, lr_scheduler, binary_csv_logger],
    verbose=1
)

# Save binary model
binary_model.save(os.path.join(model_dir, "halo_binary_model.h5"))
print("Binary model saved!")

# Visualization function
def plot_training_history(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Generate training visualizations
print("\n" + "="*50)
print("GENERATING TRAINING VISUALIZATIONS")
print("="*50)

plot_training_history(
    multi_history, 
    "Multi-Class Model Training", 
    os.path.join(model_dir, "multiclass_training_plot.png")
)

plot_training_history(
    binary_history, 
    "Binary Model Training", 
    os.path.join(model_dir, "binary_training_plot.png")
)

# Model summaries
print("\n" + "="*50)
print("MODEL SUMMARIES")
print("="*50)

print("\nMulti-class Model Summary:")
multi_class_model.summary()

print("\nBinary Model Summary:")
binary_model.summary()

# Final evaluation
print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

multi_val_loss, multi_val_acc = multi_class_model.evaluate(val_generator_multi, verbose=0)
binary_val_loss, binary_val_acc = binary_model.evaluate(val_generator_binary, verbose=0)

print(f"Multi-class Model - Validation Loss: {multi_val_loss:.4f}, Accuracy: {multi_val_acc:.4f}")
print(f"Binary Model - Validation Loss: {binary_val_loss:.4f}, Accuracy: {binary_val_acc:.4f}")

print("\n✅ Training complete! Models and visualizations saved.")
print(f"📁 Models saved in: {model_dir}")