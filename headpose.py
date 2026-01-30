import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io

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

def improved_segment_foreign_objects(image):
    """
    Enhanced segmentation using K-means clustering and multiple detection methods.
    """
    if image is None:
        print("Error: No image received.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # **1. Remove Background using rembg**
    _, buffer = cv2.imencode('.png', image)
    input_data = io.BytesIO(buffer)
    output_data = remove(input_data.read())  # Background Removal
    output_image = Image.open(io.BytesIO(output_data)).convert("RGB")  
    output_image = np.array(output_image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # **2. Apply K-means Clustering for Foreground Segmentation**
    segmented = extract_foreground_objects(output_image)

    # **3. Edge Detection**
    edges = cv2.Canny(gray, 50, 150)

    # **4. Adaptive Thresholding**
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # **5. Debugging: Check image shapes**
    print(f"Edges shape: {edges.shape}")
    print(f"Threshold shape: {thresh.shape}")
    print(f"Segmented shape: {segmented.shape}")

    # Ensure all images are the same size and type before performing bitwise operations
    if edges.shape != thresh.shape:
        thresh = cv2.resize(thresh, (edges.shape[1], edges.shape[0]))
        print("Resized threshold to match edges shape.")

    if edges.shape != segmented.shape:
        segmented = cv2.resize(segmented, (edges.shape[1], edges.shape[0]))
        print("Resized segmented to match edges shape.")

    # **6. Ensure all images are binary (0 or 255)**
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)
    _, segmented = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)

    # **7. Ensure they are uint8 (if they are not already)**
    edges = np.uint8(edges)
    thresh = np.uint8(thresh)
    segmented = np.uint8(segmented)

    # **8. Combine all detections**
    combined = cv2.bitwise_or(edges, thresh)
    combined = cv2.bitwise_or(combined, segmented)

    # **9. Morphological Processing**
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # **10. Find Contours**
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    boxes = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h)

        if (500 < area < 200000) and ((0.1 < aspect_ratio < 8) or (w > 30 and h > 5)):  # Adjusted for thin objects
            roi = gray[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi, (64, 64))
            rois.append(roi_resized)
            boxes.append((x, y, w, h))

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(f"segmented_object_{i}.jpg", roi_resized)

    return image, np.array(rois), boxes

def main():
    os.chdir("C:/Users/DEll/Downloads/dip project 1st/FYP")
    image_path = input('C:/Users/DEll/Downloads/dip project 1st/FYP/test_image2.jpg').strip()

    # Read the input image
    image = cv2.imread('C:/Users/DEll/Downloads/dip project 1st/FYP/test_image2.jpg')
    if image is None:
        print("Error: Unable to load the image.")
        return

    # Process the image
    processed_image, rois, boxes = improved_segment_foreign_objects(image)

    # Print results to console
    print("\n🔹 Segmentation Results:")
    print(f"🔹 Number of segmented objects: {len(rois)}")
    print(f"🔹 Bounding Boxes: {boxes}")

    # Display the final processed image
    cv2.imshow("Segmented Objects", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
