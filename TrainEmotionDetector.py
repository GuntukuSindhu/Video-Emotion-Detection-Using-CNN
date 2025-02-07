import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
# Define the paths
train_dir = r'C:\\Users\\sindh\\OneDrive\Desktop\\mini project\\train'
validation_dir = r'C:\\Users\\sindh\\OneDrive\Desktop\\mini project\\test'

# Check if directories exist, if not, create them
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print(f"Created directory {train_dir}.")
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)
    print(f"Created directory {validation_dir}.")

# Initialize image data generator for training and validation
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)
# Preprocess all validation images
validation_generator = validation_data_gen.flow_from_directory(
    validation_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)
# Create model structure
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))  # Assuming 7 classes
# Use the legacy Adam optimizer
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=legacy.Adam(learning_rate=0.0001, decay=1e-6),
    metrics=['accuracy']
)
# Train the neural network/model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 64
)
# Save the entire model
emotion_model.save('emotion_model.h5') 

model.summary()
