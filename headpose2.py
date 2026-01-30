import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ====================================
# 1️⃣ Foreign Object Segmentation & Classification
# ====================================

def segment_foreign_objects(image):
    """
    Uses adaptive thresholding to improve object segmentation and remove background noise.
    """
    if len(image.shape) == 3:  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  

    # ✅ Adaptive Thresholding (Removes varying background lighting conditions)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    edges = cv2.Canny(binary, 30, 100)  # Fine-tuned Canny thresholds

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois, boxes = [], []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)

        # ✅ Filters objects by size and shape
        if 3000 < area < 40000 and 0.3 < aspect_ratio < 3.5:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
            rois.append(roi_resized.astype(np.uint8))
            boxes.append((x, y, w, h))

    return np.array(rois), boxes

def build_custom_object_detection_model(input_shape=(64, 64, 1), num_classes=6):
    """
    Improved CNN model with deeper layers, Batch Normalization, and Dropout.
    """
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Stronger dropout to prevent overfitting
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def load_foreign_object_dataset(dataset_path):
    """
    Loads images from a folder structure where folder names represent labels.
    """
    images, labels = [], []
    
    # Automatically get labels from folder names
    class_mapping = {folder: idx for idx, folder in enumerate(sorted(os.listdir(dataset_path)))}
    print(f"Class Mapping: {class_mapping}")  

    for label_name, label_idx in class_mapping.items():
        label_folder = os.path.join(dataset_path, label_name)
        for filename in os.listdir(label_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(label_folder, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
                rois, _ = segment_foreign_objects(image)
                if len(rois) > 0:
                    images.append(rois[0])
                    labels.append(label_idx)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.uint8), class_mapping

def train_custom_object_model(dataset_path):
    """
    Trains a CNN model for foreign object classification with class balancing and learning rate scheduling.
    """
    print("Loading foreign object dataset...")
    images, labels, class_mapping = load_foreign_object_dataset(dataset_path)
    
    if images.size == 0:
        raise ValueError("No training images found. Check dataset folder and file naming.")

    num_classes = len(class_mapping)  
    images = images.reshape(-1, 64, 64, 1).astype(np.float32) / 255.0  
    labels = to_categorical(labels, num_classes)

    # ✅ Compute Class Weights to handle imbalanced data
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels.argmax(axis=1)),
        y=labels.argmax(axis=1)
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # ✅ Data Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )
    datagen.fit(images)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = build_custom_object_detection_model(input_shape=(64, 64, 1), num_classes=num_classes)

    # ✅ Learning Rate Scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0005 * (0.9 ** epoch))

    # ✅ Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,  # Stops training if no improvement for 6 epochs
        restore_best_weights=True
    )

    model.fit(datagen.flow(X_train, y_train, batch_size=32), 
              epochs=20,  
              validation_data=(X_test, y_test),  
              class_weight=class_weights_dict, 
              callbacks=[lr_scheduler, early_stopping])

    print("Custom object detection model training complete!")
    return model, class_mapping

# ====================================
# 2️⃣ Process and Annotate Proctoring Image
# ====================================

def process_image(image_path, object_model, class_mapping):
    """
    Improved image processing: Filters out low-confidence predictions.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  

    rois, boxes = segment_foreign_objects(image)
    
    if len(rois) > 0:
        batch_rois = np.array(rois).reshape(len(rois), 64, 64, 1).astype(np.float32) / 255.0
        predictions = object_model.predict(batch_rois)

        class_labels = list(class_mapping.keys())
        for i, (x, y, w, h) in enumerate(boxes):
            class_index = np.argmax(predictions[i])
            confidence = predictions[i][class_index]

            # ✅ Filter out low-confidence detections (threshold 0.7)
            if confidence > 0.7:
                label_text = f"{class_labels[class_index]} ({confidence:.2f})"
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(output, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Proctoring Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ====================================
# 3️⃣ Main Execution
# ====================================

if __name__ == "__main__":
    dataset_path = "C:/Users/DEll/Downloads/dip project 1st/FYP/objectdetectiondataset"
    test_image_path = "C:/Users/DEll/Downloads/dip project 1st/FYP/test_image2.jpg"

    object_model, class_mapping = train_custom_object_model(dataset_path)
    process_image(test_image_path, object_model, class_mapping)
