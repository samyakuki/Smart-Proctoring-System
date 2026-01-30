import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                   BatchNormalization, Input, Concatenate, LeakyReLU)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from rembg import remove
from PIL import Image
import io



def load_foreign_object_dataset(dataset_path):
    """
    Loads images from a folder structure where folder names represent labels.
    Returns preprocessed images, labels, and class mapping.
    """
    images = []
    labels = []
    
    # Get all subdirectories (class folders)
    class_folders = [f for f in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, f))]
    
    # Create class mapping
    class_mapping = {folder: idx for idx, folder in enumerate(sorted(class_folders))}
    print(f"Found classes: {class_mapping}")
    
    # Load images from each class folder
    for class_name, class_idx in class_mapping.items():
        class_path = os.path.join(dataset_path, class_name)
        print(f"Loading class: {class_name}")
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            
            try:
                # Read and preprocess image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not load {image_path}")
                    continue
                    
                # Resize image
                image = cv2.resize(image, (64, 64))
                
                # Add to dataset
                images.append(image)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Loaded {len(images)} images across {len(class_mapping)} classes")
    
    return images, labels, class_mapping

def create_custom_efficient_model(input_shape=(64, 64, 1), num_classes=6):
    """
    Custom efficient model with parallel paths for better feature extraction
    """
    # Input layer
    input_img = Input(shape=input_shape)
    
    # Path 1 - Fine details
    x1 = Conv2D(32, (3,3), padding='same')(input_img)
    x1 = LeakyReLU(alpha=0.1)(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(2,2)(x1)
    
    x1 = Conv2D(64, (3,3), padding='same')(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(2,2)(x1)
    
    # Path 2 - Broader features
    x2 = Conv2D(32, (5,5), padding='same')(input_img)
    x2 = LeakyReLU(alpha=0.1)(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(2,2)(x2)
    
    x2 = Conv2D(64, (5,5), padding='same')(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(2,2)(x2)
    
    # Path 3 - Context path
    x3 = Conv2D(32, (7,7), padding='same')(input_img)
    x3 = LeakyReLU(alpha=0.1)(x3)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(2,2)(x3)
    
    x3 = Conv2D(64, (7,7), padding='same')(x3)
    x3 = LeakyReLU(alpha=0.1)(x3)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(2,2)(x3)
    
    # Combine paths
    combined = Concatenate()([x1, x2, x3])
    
    # Shared layers
    x = Conv2D(128, (3,3), padding='same')(combined)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2,2)(x)
    
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_img, outputs=output)
    
    # Compile with custom learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def extract_foreground_objects(image):
    """
    Use K-means clustering for better object segmentation.
    """
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3  # Number of clusters
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def improved_segment_foreign_objects(image_path):
    """
    Enhanced segmentation using K-means clustering and multiple detection methods.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # **1. Remove Background using rembg**
    with open(image_path, 'rb') as input_file:
        input_data = input_file.read()
    output_data = remove(input_data)  # Background Removal
    output_image = Image.open(io.BytesIO(output_data)).convert("RGB")  
    output_image = np.array(output_image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # **2. Apply K-means Clustering for Foreground Segmentation**
    segmented = extract_foreground_objects(output_image)

    # **3. Edge Detection**
    edges = cv2.Canny(gray, 50, 150)

    # **4. Adaptive Thresholding**
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # **5. Combine all detections**
    combined = cv2.bitwise_or(edges, thresh)
    combined = cv2.bitwise_or(combined, segmented)

    # **6. Morphological Processing**
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # **7. Find Contours**
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    boxes = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)

        if (200 < area < 200000) and ((0.1 < aspect_ratio < 8) or (w > 30 and h > 5)):  # Adjusted for thin objects
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (64, 64))
            rois.append(roi_resized)
            boxes.append((x, y, w, h))

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(f"segmented_object_{i}.jpg", roi_resized)

    cv2.imshow("Segmented Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(rois), boxes

def train_improved_model(dataset_path):
    """
    Enhanced training procedure with data augmentation and callbacks
    """
    print("Loading dataset...")
    images, labels, class_mapping = load_foreign_object_dataset(dataset_path)
    
    if images.size == 0:
        raise ValueError("No training images found!")

    num_classes = len(class_mapping)
    images = images.reshape(-1, 64, 64, 1).astype(np.float32) / 255.0
    labels = to_categorical(labels, num_classes)

    # Enhanced data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    model = create_custom_efficient_model(input_shape=(64, 64, 1), num_classes=num_classes)
    
    # Train with batch accumulation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=60,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    print("Model training complete!")
    return model, class_mapping, history

def process_image_with_confidence(image_path, model, class_mapping):
    """
    Enhanced image processing with confidence scoring and multiple detections
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    rois, boxes = improved_segment_foreign_objects(image)
    
    if len(rois) > 0:
        batch_rois = rois.reshape(-1, 64, 64, 1).astype(np.float32) / 255.0
        predictions = model.predict(batch_rois)
        
        class_labels = list(class_mapping.keys())
        
        # Non-maximum suppression for overlapping boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            [np.max(pred) for pred in predictions],
            score_threshold=0.6,
            nms_threshold=0.3
        )
        
        # Convert indices to the correct format
        if len(indices) > 0:
            if isinstance(indices[0], np.ndarray):
                indices = indices.flatten()
            
            for i in indices:
                x, y, w, h = boxes[i]
                class_scores = predictions[i]
                
                # Get top 2 predictions if they meet threshold
                top_indices = np.argsort(class_scores)[-2:]
                for idx in reversed(top_indices):
                    if class_scores[idx] > 0.3:  # Confidence threshold
                        label_text = f"{class_labels[idx]}: {class_scores[idx]:.2f}"
                        y_offset = y - 10 - 20 * (1 - list(reversed(top_indices)).index(idx))
                        
                        # Different colors for different confidence levels
                        color = (0, 255, 0) if class_scores[idx] > 0.8 else \
                               (0, 255, 255) if class_scores[idx] > 0.6 else \
                               (0, 0, 255)
                        
                        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(output, label_text, (x, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output

def visualize_results(history):
    """
    Visualize training history
    """
    
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set paths
    dataset_path = r"C:\Users\DEll\Downloads\dip project 1st\FYP\complete_dataset"
    test_image_path = r"C:\Users\DEll\Downloads\dip project 1st\FYP\test_image2.jpg"
    
    try:
        # Train the model
        print("Starting training...")
        model, class_mapping, history = train_improved_model(dataset_path)
        
        # Visualize training results
        visualize_results(history)
        
        # Process test image
        print("Processing test image...")
        result = process_image_with_confidence(test_image_path, model, class_mapping)
        
        # Display results
        cv2.imshow("Enhanced Detection Results", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the results
        output_path = "detection_results.jpg"
        cv2.imwrite(output_path, result)
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")