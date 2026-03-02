from tqdm import tqdm
from pathlib import Path
import argparse
import tarfile
import concurrent.futures
import numpy as np
import cv2
from pymupdf import Document
import re
import io
import os
from PIL import Image
import pytesseract
import tempfile
from contextlib import contextmanager
import shutil

def ocr_page(page_data):
    """
    OCR function that works with ProcessPoolExecutor.
    Takes serialized image data.
    """
    page_index, img_bytes = page_data
    try:
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        return page_index, text
    except Exception as e:
        print(f"OCR error on page {page_index}: {e}")
        return page_index, ""

def ocr_pages(doc, dpi=300, workers=4):
    """
    Perform OCR on all pages of a PDF document.
    """
    if doc.page_count == 0:
        return [], ""
    ocr_texts = [""] * doc.page_count

    # Pre-serialize pages for ProcessPoolExecutor
    page_data_list = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_bytes = pix.tobytes("png")
        page_data_list.append((i, img_bytes))

    # Perform OCR on all pages in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(page_data_list), workers)) as executor:
        futures = {executor.submit(ocr_page, p): p[0] for p in page_data_list}
        for future in concurrent.futures.as_completed(futures):
            page_index, text = future.result()
            ocr_texts[page_index] = text

    return ocr_texts

def _detect_backbone(page_data):
    """
    Helper function that works with ProcessPoolExecutor.
    Takes serialized image data and checks for chemical backbones.
    """
    page_index, img_bytes = page_data
    np_arr = np.frombuffer(img_bytes, np.uint8)
    original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # --- 2. Remove Table Lines ---
    # This step explicitly finds and removes long horizontal and vertical lines.
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

    if detected_regions > 0:
        return page_index, True
    else:
        return page_index, False

def detect_backbone_on_pages(doc, page_indices, dpi=150, workers=4):
    """
    Checks for chemical backbones (rings or chains) on specified pages of a PDF document.
    
    Args:
        doc: The PDF document object (e.g., from fitz.open()).
        page_indices (list): A list of 0-based page indices to check.
        dpi (int): The resolution for rendering the PDF pages. 150 is a good balance.
        workers (int): The number of parallel processes to use.
    
    Returns:
        bool: True if a backbone is found on any of the specified pages, False otherwise.
    """
    page_data_list = []
    for i in page_indices:
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_bytes = pix.tobytes("png")
        page_data_list.append((i,img_bytes))

    pages_with_backbones = []
    num_workers = min(os.cpu_count() or 1, len(page_data_list), workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all pages to be processed
        futures = {executor.submit(_detect_backbone, p) for p in page_data_list}
        for future in concurrent.futures.as_completed(futures):
            if future.result()[1]:
                pages_with_backbones.append(future.result()[0])
    return pages_with_backbones

def detect_substituents(claims_text):
    """
    Return True if text contains R-group placeholders and one of several
    definition patterns indicating a Markush structure.
    """
    # Common R-group placeholders: R, R1, R2, etc.
    placeholder_patterns = [
        r"\bR\d*\b",                  # R or R1, R12...
        r"\bR-group\b",               # "R-group"
        r"\bR group\b",               # "R group"
        r"\bR\(\d+\)\b",              # R(1), R(2) in some specs
    ]

    # Definition patterns: different ways to express Markush lists or ranges
    definition_patterns = [
        r"selected from the group consisting(?: essentially of)?",  # standard
        r"selected from the group comprising",                      # comprising variant
        r"selected from(?: a group)? of",                           # shorter form
        r"wherein\s+R\d?",                                          # wherein R1 is …
        r"wherein said substituents\s+are",                         # alternative wording
        r"\bR\d?\s*to\s*R\d?\b",                                    # R1 to R5
        r"each R\d?\s*is\s*(?:a\s*)?[A-Za-z0-9,\- ]+",              # each R is …
        r"group consisting essentially of",                         # essential variant
    ]

    # Check for any placeholder
    has_placeholder = any(
        re.search(pat, claims_text, flags=re.IGNORECASE)
        for pat in placeholder_patterns
    )

    # Check for any definition phrase
    has_definition = any(
        re.search(pat, claims_text, flags=re.IGNORECASE)
        for pat in definition_patterns
    )

    return bool(has_placeholder and has_definition)

def save_pages(doc, page_indices: list, out_path):
    """
    Extracts the specified pages from a Document and writes them to out_path.
    """
    with Document() as dst:
        for i in page_indices:
            dst.insert_pdf(doc, from_page=i, to_page=i)
        dst.save(out_path)

@contextmanager
def temp_extraction_dir(prefix='extract_markush_page_'):
    """Context manager that ensures cleanup even on interruption"""
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        if os.path.exists(temp_dir):
            print(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)

def extract_valid_pdfs(tar_file, temp_dir, valid_patent_no):
    """Extract only PDFs with valid patent numbers"""
    extracted_files = []
    with tarfile.open(tar_file, 'r') as tar:
        # Get all PDF members
        pdf_members = [m for m in tar.getmembers() 
                      if m.isfile() and m.name.lower().endswith('.pdf')]
        for member in pdf_members:
            patent_no = Path(member.name).stem
            if patent_no in valid_patent_no:
                # Extract this specific file
                tar.extract(member, temp_dir)
                extracted_files.append(Path(temp_dir) / member.name)
    
    return extracted_files

def main(year: int, workers: int):
    home = Path.home()
    data_path = home / "projects/Markush/data"
    patent_no_file_path = data_path / "drug_patent_no.txt"
    with open(patent_no_file_path, 'r') as f:
        valid_patent_no = set(line.strip() for line in f.readlines())
    print(f"CPU cores available: {os.cpu_count()}")
    Path.mkdir(data_path / f'{year}/pages_with_markush', parents=True, exist_ok=True)
    tar_files = list((data_path / str(year)).glob("*.tar"))
    # tar_files = list((data_path / str(year)).glob("grant_pdf_20240102.tar"))
    tar_file_counter = 1
    
    for tar_file in tar_files:
        with temp_extraction_dir() as temp_dir:
            # Extract only valid PDFs
            pdf_files = extract_valid_pdfs(tar_file, temp_dir, valid_patent_no)
            if not pdf_files:
                print(f"No valid patents found in {tar_file.name}")
                continue

            saved_counter = 0
            no_backbone_counter = 0
            no_sub_counter = 0
            pbar = tqdm(pdf_files, total=len(pdf_files), desc=f"{tar_file.name}", unit="file")
            pbar.set_postfix(saved=saved_counter, 
                            tar_files=f'{tar_file_counter}/{len(tar_files)}',
                            no_backbone=no_backbone_counter,
                            no_sub=no_sub_counter)
            
            for pdf_path in pbar:
                patent_no = pdf_path.stem
                doc = Document(str(pdf_path))

                # 1. OCR all pages
                ocr_texts = ocr_pages(doc, workers=workers, dpi=150)

                # 2. Find pages with substituents
                pages_with_substituents = []
                for page_index, text in enumerate(ocr_texts):
                    if detect_substituents(text):
                        pages_with_substituents.append(page_index)
                if len(pages_with_substituents) == 0:
                    no_sub_counter += 1
                    pbar.set_postfix(saved=saved_counter, 
                                    tar_files=f'{tar_file_counter}/{len(tar_files)}',
                                    no_backbone=no_backbone_counter,
                                    no_sub=no_sub_counter)
                    doc.close()
                    continue

                # 3. For pages with substituents, check for backbones
                pages_with_backbones = detect_backbone_on_pages(doc, pages_with_substituents, workers=workers, dpi=150)
                if len(pages_with_backbones) == 0:
                    no_backbone_counter += 1
                    pbar.set_postfix(saved=saved_counter, 
                                    tar_files=f'{tar_file_counter}/{len(tar_files)}',
                                    no_backbone=no_backbone_counter,
                                    no_sub=no_sub_counter)
                    doc.close()
                    continue
                else:
                    output_pdf = data_path / f'{year}/pages_with_markush/{patent_no}.pdf'
                    save_pages(doc, sorted(pages_with_backbones), output_pdf)
                    saved_counter += 1
                    doc.close()
                    pbar.set_postfix(saved=saved_counter,
                                    tar_files=f'{tar_file_counter}/{len(tar_files)}',
                                    no_backbone=no_backbone_counter,
                                    no_sub=no_sub_counter)
            
        tar_file_counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract claim sections from patent PDFs which contains Markush structure.")
    parser.add_argument("--year", type=int, required=True, help="Year of the patents to process.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for parallel processing.")
    args = parser.parse_args()
    main(args.year, args.workers)