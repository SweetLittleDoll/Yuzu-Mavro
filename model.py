import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(inputs, filters, kernel_size=3, padding='same'):
    """Convolutional block with batch normalization and ReLU activation"""
    x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def encoder_block(inputs, filters, kernel_size=3, padding='same'):
    """Encoder block with two convolutional layers and max pooling"""
    x = conv_block(inputs, filters, kernel_size, padding)
    x = conv_block(x, filters, kernel_size, padding)
    skip = x
    x = layers.MaxPooling2D((2, 2))(x)
    return x, skip

def decoder_block(inputs, skip_features, filters, kernel_size=3, padding='same'):
    """Decoder block with upsampling, concatenation, and two convolutional layers"""
    x = layers.UpSampling2D((2, 2))(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters, kernel_size, padding)
    x = conv_block(x, filters, kernel_size, padding)
    return x

def build_unet(input_shape=(128, 256, 3), num_classes=5):
    """Build U-Net model for semantic segmentation"""
    inputs = layers.Input(input_shape)
    
    # Encoder path
    x1, skip1 = encoder_block(inputs, 64)
    x2, skip2 = encoder_block(x1, 128)
    x3, skip3 = encoder_block(x2, 256)
    x4, skip4 = encoder_block(x3, 512)
    
    # Bridge
    x5 = conv_block(x4, 1024)
    x5 = conv_block(x5, 1024)
    
    # Decoder path
    x6 = decoder_block(x5, skip4, 512)
    x7 = decoder_block(x6, skip3, 256)
    x8 = decoder_block(x7, skip2, 128)
    x9 = decoder_block(x8, skip1, 64)
    
    # Output layer - no activation (logits for sparse_categorical_crossentropy)
    outputs = layers.Conv2D(num_classes, 1, activation=None)(x9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_simple_unet(input_shape=(128, 256, 3), num_classes=5):
    """Build a simpler U-Net model for faster training"""
    inputs = layers.Input(input_shape)
    
    # Encoder path
    x1, skip1 = encoder_block(inputs, 32)
    x2, skip2 = encoder_block(x1, 64)
    x3, skip3 = encoder_block(x2, 128)
    
    # Bridge
    x4 = conv_block(x3, 256)
    x4 = conv_block(x4, 256)
    
    # Decoder path
    x5 = decoder_block(x4, skip3, 128)
    x6 = decoder_block(x5, skip2, 64)
    x7 = decoder_block(x6, skip1, 32)
    
    # Output layer - no activation (logits for sparse_categorical_crossentropy)
    outputs = layers.Conv2D(num_classes, 1, activation=None)(x7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
