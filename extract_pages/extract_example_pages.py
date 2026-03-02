from pymupdf import Document
import pytesseract
from pathlib import Path
import argparse
import os
import tempfile
from contextlib import contextmanager
import shutil
import tarfile
from tqdm import tqdm
from PIL import Image
import concurrent.futures
import io
import re


@contextmanager
def temp_extraction_dir(prefix='extract_example_pages_'):
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

def save_pages(doc, page_indices: list, out_path):
    """
    Extracts the specified pages from a Document and writes them to out_path.
    """
    with Document() as dst:
        for i in page_indices:
            dst.insert_pdf(doc, from_page=i, to_page=i)
        dst.save(out_path)

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

def main(year: int, workers: int):
    home = Path.home()
    data_path = home / "projects/Markush/data"
    print(f"CPU cores available: {os.cpu_count()}")
    Path.mkdir(data_path / f'{year}/pages_with_example', parents=True, exist_ok=True)

    EXAMPLE_PREP_PATTERN = re.compile(
    r'example\s+\d+[a-z]?\s*[:.\-\s]\s*(preparation|synthesis)\s+of',
    re.IGNORECASE)

    # First pick up patent with Markush structures
    pdf_files = list((data_path / str(year) / 'claimed_markush').glob("*.png"))
    valid_patent_no = [pdf_file.stem for pdf_file in pdf_files]
    tar_files = list((data_path / str(year)).glob("*.tar"))
    tar_file_counter = 1
    for tar_file in tar_files:
        with temp_extraction_dir() as temp_dir:
            # Extract only valid PDFs
            full_pdf_files = extract_valid_pdfs(tar_file, temp_dir, valid_patent_no)
            if not full_pdf_files:
                print(f"No valid patents found in {tar_file.name}")
                continue

            saved_counter = 0
            pbar = tqdm(full_pdf_files, total=len(full_pdf_files), desc=f"{tar_file.name}", unit="file")
            pbar.set_postfix(saved=saved_counter, 
                            tar_files=f'{tar_file_counter}/{len(tar_files)}',)
            
            for pdf_path in pbar:
                patent_no = pdf_path.stem
                doc = Document(str(pdf_path))
                ocr_text_list = ocr_pages(doc, workers=workers, dpi=300)
                pages_with_example = []
                for page_index, page_text in enumerate(ocr_text_list):
                    if EXAMPLE_PREP_PATTERN.search(page_text):
                        pages_with_example.append(page_index)
                if len(pages_with_example) > 0:
                    output_pdf = data_path / f'{year}/pages_with_example' / f"{patent_no}.pdf"
                    save_pages(doc, sorted(pages_with_example), output_pdf)
                    doc.close()
                    saved_counter += 1
                    pbar.set_postfix(saved=saved_counter,
                                    tar_files=f'{tar_file_counter}/{len(tar_files)}',)
            tar_file_counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract pages which contain example and preparation/synthesis.")
    parser.add_argument('--year', type=int, required=True, help='Year of the patents to process')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for OCR')
    args = parser.parse_args()
    main(args.year, args.workers)
