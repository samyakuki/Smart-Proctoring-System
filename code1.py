import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, Input, LeakyReLU, GlobalAveragePooling2D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def create_model(input_shape=(64, 64, 3), num_classes=12):
    inp = Input(shape=input_shape)
    x = Conv2D(32, 3, padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.1)(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_with_watershed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    return markers

def extract_watershed_rois(image, markers):
    rois, boxes = [], []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.uint8(markers == label)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if 300 < area < 200000:
                roi = image[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (64, 64))
                rois.append(roi_resized)
                boxes.append((x, y, w, h))
    return np.array(rois), boxes

def draw_detections(image, boxes, predictions, class_labels):
    for i, (x, y, w, h) in enumerate(boxes):
        label_id = np.argmax(predictions[i])
        confidence = predictions[i][label_id]
        if confidence > 0.3:
            label = f"{class_labels[label_id]}: {confidence:.2f}"
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def process_frame(image, model, class_labels):
    markers = preprocess_with_watershed(image)
    rois, boxes = extract_watershed_rois(image, markers)
    if len(rois) == 0:
        return image
    input_rois = rois.astype(np.float32) / 255.0
    predictions = model.predict(input_rois, verbose=0)
    return draw_detections(image, boxes, predictions, class_labels)

if __name__ == "__main__":
    model = create_model()
    try:
        model.load_weights("best_model.keras")
    except:
        print("[WARNING] Could not load pretrained weights.")

    class_labels = ['inappropriate', 'notebook', 'pen', 'phone', 'ipad', 'bottle',
                    'left', 'right', 'down', 'upwards', 'straight', 'watch']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = process_frame(frame.copy(), model, class_labels)
        cv2.imshow("Proctoring Detector", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()