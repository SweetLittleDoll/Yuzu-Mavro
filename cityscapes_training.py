# Cityscapes Semantic Segmentation Challenge
# This script implements a U-Net model for semantic segmentation on the custom Cityscapes dataset.
#
# Dataset Information:
# - Images: 1000 RGB images (128x256x3)
# - Masks: 1000 segmentation masks (128x256) with 5 classes
# - Classes: Background (0), Road (1), Sidewalk (2), Pedestrian (3), Vehicle (4)
# - Split: 70% train, 15% validation, 15% test

# ============================================================================
# CELL 1: Setup and Imports
# ============================================================================
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# CELL 2: Load and Preprocess Dataset
# ============================================================================
from utils import load_dataset, preprocess_data, split_dataset

# Load dataset
data_dir = './data_path_for_mavro_challenge'  # Update this path
images, masks = load_dataset(data_dir)

# Preprocess data
images_processed, masks_processed = preprocess_data(images, masks)

# Split dataset
(train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_dataset(
    images_processed, masks_processed
)

# ============================================================================
# CELL 3: Visualize Sample Data
# ============================================================================
def visualize_sample(image, mask, title="Sample"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"{title} - Image")
    axes[0].axis('off')
    
    # Segmentation mask (now integer values 0-4)
    axes[1].imshow(mask, cmap='tab10')
    axes[1].set_title(f"{title} - Mask")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize a few samples
for i in range(3):
    visualize_sample(train_images[i], train_masks[i], f"Sample {i+1}")

# ============================================================================
# CELL 4: Build Model Architecture
# ============================================================================
from model import build_simple_unet

# Build model
model = build_simple_unet(input_shape=(128, 256, 3), num_classes=5)

# Compile model with standard Keras loss and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Numerically stable with logits
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanIoU(num_classes=5, name='mean_iou'),  # Official IoU metric
        tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')
    ]
)

model.summary()

# ============================================================================
# CELL 5: Training Configuration
# ============================================================================
# Training parameters
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_mean_iou',  # Monitor IoU instead of dice
    save_best_only=True,
    mode='max'
)

# ============================================================================
# CELL 6: Train the Model
# ============================================================================
# Train the model
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# ============================================================================
# CELL 7: Save Final Model
# ============================================================================
# Save the final trained model
model.save('model.keras')
print("Model saved as 'model.keras'")

# ============================================================================
# CELL 8: Training Results Visualization
# ============================================================================
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training vs Validation IoU (REQUIRED for challenge)
    axes[0, 0].plot(history.history['mean_iou'], label='Training IoU')
    axes[0, 0].plot(history.history['val_mean_iou'], label='Validation IoU')
    axes[0, 0].set_title('Training vs Validation IoU')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('IoU Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training vs Validation Loss (REQUIRED for challenge)
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Training vs Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training vs Validation Accuracy
    axes[1, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1, 0].set_title('Training vs Validation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Training vs Validation Sparse Categorical Accuracy
    axes[1, 1].plot(history.history['sparse_categorical_accuracy'], label='Training Sparse Categorical Accuracy')
    axes[1, 1].plot(history.history['val_sparse_categorical_accuracy'], label='Validation Sparse Categorical Accuracy')
    axes[1, 1].set_title('Training vs Validation Sparse Categorical Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)

# ============================================================================
# CELL 9: Model Evaluation on Test Set
# ============================================================================
# Evaluate on test set
test_results = model.evaluate(test_images, test_masks, verbose=1)
test_loss, test_acc, test_iou, test_sparse_acc = test_results

print(f"\nTest Set Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Mean IoU: {test_iou:.4f}")
print(f"Sparse Categorical Accuracy: {test_sparse_acc:.4f}")

# ============================================================================
# CELL 10: Sample Predictions Visualization
# ============================================================================
def visualize_predictions(model, images, masks, num_samples=5):
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image = images[idx]
        true_mask = masks[idx]
        
        # Make prediction
        pred_logits = model.predict(np.expand_dims(image, axis=0))[0]
        pred_mask = np.argmax(pred_logits, axis=-1)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title(f"Sample {i+1} - Image")
        axes[0].axis('off')
        
        axes[1].imshow(true_mask, cmap='tab10')
        axes[1].set_title(f"Sample {i+1} - Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='tab10')
        axes[2].set_title(f"Sample {i+1} - Prediction")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

# Visualize predictions on test set
visualize_predictions(model, test_images, test_masks)

# ============================================================================
# CELL 11: Class-wise Performance Analysis
# ============================================================================
def calculate_class_metrics(y_true, y_pred):
    class_names = ['Background', 'Road', 'Sidewalk', 'Pedestrian', 'Vehicle']
    
    # Calculate IoU for each class
    class_iou = []
    for class_id in range(5):
        intersection = np.logical_and(y_true == class_id, y_pred == class_id).sum()
        union = np.logical_or(y_true == class_id, y_pred == class_id).sum()
        iou = intersection / (union + 1e-6)
        class_iou.append(iou)
    
    # Plot class-wise IoU
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_iou, color='skyblue')
    plt.title('Class-wise IoU Scores')
    plt.xlabel('Classes')
    plt.ylabel('IoU Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, iou in zip(bars, class_iou):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{iou:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return class_iou

# Get predictions for test set
test_predictions = model.predict(test_images)
test_pred_masks = np.argmax(test_predictions, axis=-1)

# Calculate and visualize class-wise metrics
class_iou_scores = calculate_class_metrics(test_masks, test_pred_masks)

print("\nClass-wise IoU Scores:")
for i, (class_name, iou) in enumerate(zip(['Background', 'Road', 'Sidewalk', 'Pedestrian', 'Vehicle'], class_iou_scores)):
    print(f"{class_name}: {iou:.4f}")

# ============================================================================
# CELL 12: Summary and Final Results
# ============================================================================
print("=== FINAL MODEL PERFORMANCE ===")
print(f"Training Epochs: {len(history.history['loss'])}")
print(f"Best Validation IoU: {max(history.history['val_mean_iou']):.4f}")
print(f"Best Validation Loss: {min(history.history['val_loss']):.4f}")
print(f"Test Set IoU: {test_iou:.4f}")
print(f"Test Set Loss: {test_loss:.4f}")
print(f"Test Set Accuracy: {test_acc:.4f}")
print("\nModel saved as 'model.keras'")
print("Training completed successfully!")
