import cv2
import numpy as np
from pathlib import Path
from pymupdf import Document
from pathlib import Path

def identify_chemical_backbones(image_path):
    """
    Identifies chemical chains and rings in an image, now with table line removal.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        A tuple containing:
        - original_image: The original image with detected structures highlighted.
        - processing_steps (dict): A dictionary of images from each processing step.
    """
    # --- 1. Load Image and Pre-process ---
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # --- 2. Remove Table Lines ---
    # This step explicitly finds and removes long horizontal and vertical lines.
    
    # Create a copy to work on
    no_tables_mask = binary_image.copy()
    
    # Define kernels for horizontal and vertical line detection
    # The length (e.g., 50) should be adjusted based on the expected minimum
    # length of a table line in your documents.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    
    # Detect horizontal lines
    detected_horizontal = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines
    detected_vertical = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine detected lines into a single mask and remove them from the image
    table_lines_mask = detected_horizontal + detected_vertical
    no_tables_mask = cv2.subtract(no_tables_mask, table_lines_mask)


    # --- 3. Filter Main Texts to Keep Potential Rings and Chains ---
    # Now run the component analysis on the table-free image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(no_tables_mask, 4, cv2.CV_32S)

    filtered_mask = np.zeros_like(no_tables_mask)
    
    MIN_AREA_NOISE = 15
    MAX_CHAR_HEIGHT = 40
    MIN_CHAR_HEIGHT = 5
    MAX_ASPECT_RATIO = 5
    MIN_ASPECT_RATIO = 0.1
    MIN_SOLIDITY = 0.25

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA_NOISE:
            continue
        
        aspect_ratio = w / float(h)
        component_mask = (labels == i).astype("uint8") * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0: continue
            
        contour = contours[0]
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0

        is_text_like = (
            MIN_CHAR_HEIGHT < h < MAX_CHAR_HEIGHT and
            MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and
            solidity > MIN_SOLIDITY
        )

        if not is_text_like:
            filtered_mask[component_mask == 255] = 255

    # --- 4. Region Grouping ---
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = original_image.copy()
    detected_regions = 0

    # --- 5. Line and Corner Detection to Verify Structures ---
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 50 or h < 50: continue

        # Use the original binary image (not the table-free one) for ROI analysis
        # to ensure bonds that were close to table lines are still detected correctly.
        roi = binary_image[y:y+h, x:x+w]
        
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=5)
        num_lines = len(lines) if lines is not None else 0

        corners = cv2.goodFeaturesToTrack(roi, maxCorners=100, qualityLevel=0.01, minDistance=10)
        num_corners = len(corners) if corners is not None else 0
        
        # Additional check: A real chemical structure should not be excessively wide or tall.
        # This helps filter out any remaining border-like artifacts.
        roi_aspect_ratio = w / float(h)
        if roi_aspect_ratio > 10 or roi_aspect_ratio < 0.1:
            continue

        if num_lines > 10 and num_corners > 10:
            detected_regions += 1
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"Found {detected_regions} potential chemical structure(s).")
    
    processing_steps = {
        "1_gray": gray_image,
        "2_binary": binary_image,
        "3_table_lines_removed": no_tables_mask,
        "4_filtered_components": filtered_mask,
        "5_dilated_regions": dilated_mask,
        "6_final_output": output_image
    }

    return output_image, processing_steps

if __name__ == '__main__':
    # --- Configuration ---
    home = Path.home()
    PDF_PATH = home / "projects/Markush/data/2024/pages_with_markush/12178875.pdf"  # <--- CHANGE THIS
    PAGE_NUMBER = 0  # <--- CHANGE THIS to a page with chemical rings
    DPI = 150
    IMAGE_PATH = home / "projects/Markush/extract_pages/demo/cv/sample_page_150dpi.png"
    # -------------------
    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        print(f"Error: PDF not found at {pdf_file}")
    else:
        doc = Document(str(pdf_file))
        if 0 <= PAGE_NUMBER < doc.page_count:         # Ensure the page number is valid
            page = doc.load_page(PAGE_NUMBER)
            pix = page.get_pixmap(dpi=DPI) # Get pixmap at the desired DPI
            
            pix.save(IMAGE_PATH) # Save the image
        else:
            print(f"Error: Invalid page number. The document has {doc.page_count} pages.")
            raise ValueError("Invalid page number.")
        doc.close()

    final_image, steps = identify_chemical_backbones(IMAGE_PATH)
    if steps:
        for step in steps:
            cv2.imwrite(home / f'projects/Markush/extract_pages/demo/cv/{step}.png', steps[step])