import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
IMG_SIZE = 64  # Image size for resizing
BATCH_SIZE = 32
EPOCHS = 10

# Define class names based on the dataset structure
CLASS_NAMES = [
    'palm', 'l', 'fist', 'fist_moved', 'thumb', 
    'index', 'ok', 'palm_moved', 'c', 'down'
]
NUM_CLASSES = len(CLASS_NAMES)

def load_data(data_dir):
    """
    Load images and labels from the dataset directory
    """
    images = []
    labels = []
    total_images = 0
    
    print(f"Searching for dataset in: {data_dir}")
    
    # First, verify the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    print("Available directories in dataset folder:")
    for item in os.listdir(data_dir):
        print(f"- {item}")
    
    print("\nLoading dataset...")
    
    # Get list of all subdirectories (00, 01, 02, etc.)
    subdirs = [d for d in os.listdir(data_dir) 
              if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
    subdirs.sort()
    
    if not subdirs:
        raise ValueError("No numbered subdirectories found in the dataset directory.")
    
    # Map class names to their corresponding directory prefixes (e.g., 'palm' -> '01_palm')
    class_to_dir = {}
    for idx, class_name in enumerate(CLASS_NAMES, 1):
        class_to_dir[class_name] = f"{idx:02d}_{class_name}"
    
    # Process each class
    for class_idx, class_name in enumerate(CLASS_NAMES):
        print(f"\nProcessing class: {class_name} (looking for directories matching: {class_to_dir[class_name]})")
        class_count = 0
        
        # Look for this class in each subdirectory
        for subdir in subdirs:
            class_dir = os.path.join(data_dir, subdir, class_to_dir[class_name])
            
            if not os.path.exists(class_dir):
                print(f"  Directory not found: {class_dir}")
                continue
                
            print(f"  Found class directory: {class_dir}")
            
            # Get all PNG files in this directory
            try:
                png_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.png')]
                print(f"  Found {len(png_files)} images")
                
                for file in tqdm(png_files, desc=f"  Loading {class_name} from {subdir}"):
                    try:
                        img_path = os.path.join(class_dir, file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        if img is None:
                            print(f"Warning: Could not read image {img_path}")
                            continue
                            
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = img / 255.0  # Normalize pixel values
                        
                        images.append(img)
                        labels.append(class_idx)
                        class_count += 1
                        total_images += 1
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        
            except Exception as e:
                print(f"Error reading directory {class_dir}: {e}")
        
        if class_count == 0:
            print(f"Warning: No images found for class '{class_name}'")
    
    print(f"\nTotal images loaded: {total_images}")
    if total_images == 0:
        raise ValueError("No images were loaded. Please check the dataset path and structure.")
    
    return np.array(images), np.array(labels)

def create_model(input_shape, num_classes):
    """
    Create a CNN model for hand gesture recognition
    """
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Dataset path - using the D: drive location
    data_dir = r"D:\leapGestRecog\leapGestRecog"
    
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load the dataset
    print("Loading dataset from:", data_dir)
    X, y = load_data(data_dir)
    
    # Reshape data for CNN (add channel dimension)
    X = X.reshape((-1, IMG_SIZE, IMG_SIZE, 1))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create data generators with data augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=BATCH_SIZE
    )
    
    validation_generator = train_datagen.flow(
        X_test, y_test, batch_size=BATCH_SIZE
    )
    
    # Create and compile the model
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    model = create_model(input_shape, NUM_CLASSES)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.h5', 
                save_best_only=True,
                save_weights_only=False
            )
        ]
    )
    
    # Plot and save training history
    plot_training_history(history)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save the final model
    model.save('models/hand_gesture_model.h5')
    print("\nModel saved to 'models/hand_gesture_model.h5'")
    
    # Save class names for later use
    with open('models/class_names.txt', 'w') as f:
        for class_name in CLASS_NAMES:
            f.write(f"{class_name}\n")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
