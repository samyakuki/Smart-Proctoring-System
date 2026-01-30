import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, Input, LeakyReLU, GlobalAveragePooling2D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from rembg import remove
from PIL import Image
import io

# C:\Users\DEll\Downloads\dip project 1st\FYP\ezyZip

def load_foreign_object_dataset(dataset_path):
    images, labels = [], []
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    class_mapping = {folder: idx for idx, folder in enumerate(sorted(class_folders))}
    for class_name, class_idx in class_mapping.items():
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (64, 64))
            images.append(image)
            labels.append(class_idx)
    return np.array(images), np.array(labels), class_mapping

def create_custom_efficient_model(input_shape=(64, 64, 1), num_classes=12):
    input_img = Input(shape=input_shape)
    def conv_block(x, filters, kernel_size):
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D(2, 2)(x)
        return x
    x = conv_block(input_img, 32, 3)
    x = conv_block(x, 64, 3)
    x = conv_block(x, 128, 3)
    x = conv_block(x, 256, 3)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=output)
    model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def extract_foreground_objects(image, k_clusters=3, downscale=2):
    # Downscale image for faster k-means
    small_img = cv2.resize(image, (image.shape[1] // downscale, image.shape[0] // downscale))
    
    # Reshape image to 1D array of pixels and convert to float32
    Z = small_img.reshape((-1, 3)).astype(np.float32)
    
    # Apply K-means clustering to pixel colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Map clusters back to image
    segmented = centers[labels.flatten()].reshape(small_img.shape).astype(np.uint8)
    
    # Upscale segmented image to original size
    return cv2.resize(segmented, (image.shape[1], image.shape[0]))


def improved_segment_foreign_objects(image_np, image_path=None, use_rembg=False):
    # Optional background removal
    if use_rembg and image_path:
        with open(image_path, 'rb') as f:
            data = f.read()
        rembg_img = Image.open(io.BytesIO(remove(data))).convert("RGB")
        image_np = cv2.cvtColor(np.array(rembg_img), cv2.COLOR_RGB2BGR)

    # Preprocess image
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Optional: CLAHE for uniform lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply Gaussian blur to smooth
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding for clean binary mask
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology to close gaps
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    _, morphed = cv2.threshold(morphed, 127, 255, cv2.THRESH_BINARY)

    # Save debug image
    cv2.imwrite("debug_combined_mask.jpg", morphed)
    print("[DEBUG] Saved debug_combined_mask.jpg")
    print("[DEBUG] Unique values in mask:", np.unique(morphed))

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_contour_img = image_np.copy()
    cv2.drawContours(debug_contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug_contours.jpg", debug_contour_img)
    print("[DEBUG] Saved debug_contours.jpg with all raw contours")

    rois, boxes = [], []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0

        print(f"[DEBUG] Contour #{i} -> Area: {area}, AR: {aspect_ratio:.2f}, Solidity: {solidity:.2f}")

        if area > 200 and solidity > 0.3 and 0.5 < aspect_ratio < 5:
            roi = gray[y:y + h, x:x + w]
            rois.append(cv2.resize(roi, (64, 64)))
            boxes.append((x, y, w, h))

    return np.array(rois), boxes

def train_improved_model(dataset_path):
    images, labels, class_mapping = load_foreign_object_dataset(dataset_path)
    if images.size == 0:
        raise ValueError("No training images found.")
    images = images.reshape(-1, 64, 64, 1).astype(np.float32) / 255.0
    labels = to_categorical(labels, num_classes=len(class_mapping))
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 32
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    model = create_custom_efficient_model(input_shape=(64, 64, 1), num_classes=len(class_mapping))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=60, callbacks=callbacks, verbose=1)
    return model, class_mapping, history

def process_image_with_confidence(image_path, model, class_mapping):
    image = cv2.imread(image_path)
    if image is None:
        return None
    output = image.copy()
    rois, boxes = improved_segment_foreign_objects(image, image_path=None)  # rembg off for now
    print(f"Number of ROIs: {len(rois)}")
    print(f"Number of Boxes: {len(boxes)}")
    if len(rois) > 0:
        batch_rois = rois.reshape(-1, 64, 64, 1).astype(np.float32) / 255.0
        predictions = model.predict(batch_rois, batch_size=16, verbose=0)
        class_labels = list(class_mapping.keys())
        for i, box in enumerate(boxes):
            x, y, w, h = box
            class_scores = predictions[i]
            top_idx = np.argmax(class_scores)
            confidence = class_scores[top_idx]
            if confidence > 0.3:
                label = f"{class_labels[top_idx]}: {confidence:.2f}"
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return output

def visualize_results(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_path = r"C:\Users\DEll\Downloads\dip project 1st\FYP\complete_dataset"
    test_image_path = r"C:\Users\DEll\Downloads\dip project 1st\FYP\test_image2.jpg"
    try:
        print("Training model...")
        model, class_mapping, history = train_improved_model(dataset_path)
        visualize_results(history)
        print("Processing test image...")
        result = process_image_with_confidence(test_image_path, model, class_mapping)
        if result is not None:
            cv2.imwrite("detection_results.jpg", result)
            print("Result saved to detection_results.jpg")
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title("Detection Results")
            plt.axis("off")
            plt.show()
        else:
            print("No result image generated.")
    except Exception as e:
        print(f"Error: {e}")