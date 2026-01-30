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

def create_custom_efficient_model(input_shape=(64, 64, 1), num_classes=6):
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

def extract_foreground_objects(image):
    small_img = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    Z = small_img.reshape((-1, 3))
    Z = np.float32(Z)
    _, labels, centers = cv2.kmeans(Z, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented = centers[labels.flatten()].reshape(small_img.shape)
    return cv2.resize(np.uint8(segmented), (image.shape[1], image.shape[0]))

def improved_segment_foreign_objects(image_np, image_path=None):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    if image_path:
        with open(image_path, 'rb') as f:
            data = f.read()
        image_np = np.array(Image.open(io.BytesIO(remove(data))).convert("RGB"))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    segmented = extract_foreground_objects(image_np)
    segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    combined = cv2.bitwise_or(cv2.bitwise_or(edges, thresh), segmented_gray)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("debug_combined_mask.jpg", combined)
    print("[DEBUG] Saved combined mask to debug_combined_mask.jpg")
    rois, boxes = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ar = w / float(h)
        print(f"[DEBUG] Box: ({x}, {y}, {w}, {h}) Area: {area}, AR: {ar:.2f}")
    
        if area > 100:  # lower threshold to test
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
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, save_format='keras')
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks, verbose=1)
    return model, class_mapping, history

# def process_image_with_confidence(image_path, model, class_mapping):
#     image = cv2.imread(image_path)
#     if image is None:
#         return None
#     output = image.copy()
#     rois, boxes = improved_segment_foreign_objects(image, image_path=image_path)
#     if len(rois) > 0:
#         batch_rois = rois.reshape(-1, 64, 64, 1).astype(np.float32) / 255.0
#         predictions = model.predict(batch_rois, batch_size=16, verbose=0)
#         class_labels = list(class_mapping.keys())
#         confidences = [float(np.max(pred)) for pred in predictions]
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)
#         if len(indices) > 0:
#             indices = indices.flatten() if isinstance(indices[0], np.ndarray) else indices
#             for i in indices:
#                 x, y, w, h = boxes[i]
#                 class_scores = predictions[i]
#                 top_indices = np.argsort(class_scores)[-2:]
#                 for idx in reversed(top_indices):
#                     if class_scores[idx] > 0.3:
#                         label = f"{class_labels[idx]}: {class_scores[idx]:.2f}"
#                         y_offset = y - 10 - 20 * (1 - list(reversed(top_indices)).index(idx))
#                         color = (0, 255, 0) if class_scores[idx] > 0.8 else (0, 255, 255) if class_scores[idx] > 0.6 else (0, 0, 255)
#                         cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
#                         cv2.putText(output, label, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     return output

def process_image_with_confidence(image_path, model, class_mapping):
    image = cv2.imread(image_path)
    if image is None:
        return None
    output = image.copy()
    rois, boxes = improved_segment_foreign_objects(image, image_path=image_path)
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

            if confidence > 0.3:  # Threshold to display
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




# Save to file


if __name__ == "__main__":
    dataset_path = r"C:\Users\DEll\Downloads\dip project 1st\FYP\ezyZip"
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
        else:
            print("No result image generated.")
    except Exception as e:
        print(f"Error: {e}")
        

