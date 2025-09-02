import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_dataset(data_dir):
    """Load the challenge images and masks from .npy files"""
    images_file = f"{data_dir}/challenge_images.npy"
    masks_file = f"{data_dir}/challenge_masks.npy"
    
    images = np.load(images_file)
    masks = np.load(masks_file)
    
    print(f"Loaded images shape: {images.shape}")
    print(f"Loaded masks shape: {masks.shape}")
    
    return images, masks

def preprocess_data(images, masks):
    """Preprocess images and masks for training"""
    # Normalize images to [0, 1]
    images = images.astype(np.float32) / 255.0
    
    # Keep masks as integers (0-4) - no one-hot encoding needed
    # This is more memory efficient and aligns with Keras standards
    masks = masks.astype(np.uint8)
    
    return images, masks

def split_dataset(images, masks, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train, validation, and test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: separate test set
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        images, masks, test_size=test_ratio, random_state=42
    )
    
    # Second split: separate validation from train
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=val_ratio_adjusted, random_state=42
    )
    
    print(f"Train set: {train_images.shape[0]} samples")
    print(f"Validation set: {val_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")
    
    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient for one-hot encoded predictions"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function for one-hot encoded predictions"""
    return 1 - dice_coefficient(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1e-6):
    """Calculate IoU score for one-hot encoded predictions"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def create_data_generator(images, masks, batch_size=8):
    """Create a data generator for training"""
    num_samples = len(images)
    
    while True:
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = images[batch_indices]
            batch_masks = masks[batch_indices]
            yield batch_images, batch_masks
