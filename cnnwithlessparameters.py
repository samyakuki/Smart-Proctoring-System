import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io
def improved_segment_foreign_objects(image):
    """
    Enhanced segmentation with multiple detection methods
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None, None
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    
    
    with open(gray, 'rb') as input_file:
        input_data = input_file.read()

    # Remove the background using rembg
    output_data = remove(input_data)

    # Convert the output data into an image
    output_image = Image.open(io.BytesIO(output_data))
    
    # 2. Edge detection
    edges = cv2.Canny(gray, 30, 150)
    
    # 3. Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine detection methods
    combined = cv2.bitwise_or(output_image, edges)
    combined = cv2.bitwise_or(combined, thresh)
    
    # Enhance detection
    # kernel = np.ones((3,3), np.uint8)
    # combined = cv2.dilate(combined, kernel, iterations=1)
    # combined = cv2.erode(combined, kernel, iterations=1)
    
    # contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # rois = []
    # boxes = []
    
    # for i, contour in enumerate(contours):
    #     x, y, w, h = cv2.boundingRect(contour)
    #     area = w * h
    #     aspect_ratio = w / float(h)
        
    #     if 800 < area < 50000 and 0.2 < aspect_ratio < 5:
    #         roi = gray[y:y+h, x:x+w]
    #         roi_resized = cv2.resize(roi, (64, 64))
    #         rois.append(roi_resized)
    #         boxes.append((x, y, w, h))
            
    #         # Draw bounding box on original image
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    #         # Save segmented regions
    #         cv2.imwrite(f"segmented_object_{i}.jpg", roi_resized)
    
    # Display the processed image
    cv2_imshow(combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    image_path = '/content/drive/MyDrive/dataset/ipad/Movie-on-11-01-25-at-1_29-PM_mov-0004_jpg.rf.77749cbb00d48100692fc57aae9fc90d.jpg'
    output_rois, output_boxes = improved_segment_foreign_objects(image_path)
    
  