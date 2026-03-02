import numpy as np
import pandas as pd
import requests
import os
from typing import Tuple
import concurrent.futures
from pymupdf import Document
import argparse
import nltk
nltk.download('punkt', quiet=True)
import pytesseract
import warnings
from py2opsin import py2opsin
warnings.filterwarnings("ignore", module="py2opsin")
from tqdm import tqdm
from pathlib import Path
home = Path.home()
data_path = home / "projects/Markush/data"
OLLAMA_URL = "http://localhost:11434/api/generate"

def ocr_page(page_data):
    """
    OCR function that works with ProcessPoolExecutor.
    Takes serialized image data.
    """
    page_index, img_array = page_data
    text = pytesseract.image_to_string(img_array)
    return page_index, text


def extract_iupac_using_llm(text_input: str, model_name: str) -> str:
    prompt = f"""
You are analyzing a page from a chemical patent document. The text below is OCR-extracted and may contain noise.

Your task is to extract **only the full IUPAC names of specific, well-defined example compounds** that are:
- Explicitly prepared, synthesized, or exemplified (e.g., "Example 1", "Compound 5a", "Synthesization of", "Preparation of", "embodiment")
- Part of the claimed Markush structure with specific substituents filled in (e.g., R¹ = methyl, R² = phenyl)

### ✅ DO extract:
- Fully substituted IUPAC names with specific groups (e.g., "7-chloro-3-methyl-1,3-dihydro-2H-pyrrolo[2,1-c][1,4]benzodiazepin-2-one")
- Names that include locants and specific substituents (e.g., "N-(4-cyanophenyl)-2-hydroxybenzamide")

### ❌ DO NOT extract:
- Generic scaffold names (e.g., "pyrrolobenzodiazepine", "quinazoline", "benzimidazole")
- Partial or unsubstituted ring systems without specific substitution patterns
- Terms ending in or containing: "derivative", "analogue", "compound of formula I", "as defined herein"
- Generic substituents (e.g., "alkyl", "aryl", "halogen")
- Reagents, solvents, monomers, surfactants, intermediates, or building blocks
- Trade names, abbreviations (e.g., DMSO), molecular formulas (e.g., C₁₀H₁₄N₂), or Markush variables (e.g., R¹, R²)
- Reference compounds or prior art

### 🚫 Critical Rule:
> If the compound name lacks specific substitution details (positions and identities), it is **too generic** and must be excluded.
> For example:
> - ❌ "pyrrolobenzodiazepine" → too generic
> - ✅ "8-bromo-3-(pyridin-3-ylmethyl)-1,3-dihydro-2H-pyrrolo[2,1-c][1,4]benzodiazepin-2-one" → specific

Output only the full IUPAC names, one per line.
If the same compound appears multiple times, output it only once.
If no valid example compound IUPAC name is found, return an empty string.

Text:
{text_input.strip()}
"""
     
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    iupac_name = result.get("response", "").strip()
    return iupac_name if iupac_name else ""


def iupac_to_smile(iupac_names: list) -> Tuple[list[tuple[str, str]], int]:
    compounds = []
    fail_counter = 0
    for name in iupac_names:
        smile = py2opsin(name)
        if smile:
            compounds.append((name, smile))
        else:
            fail_counter += 1
    return compounds, fail_counter


def main(year: int, patent_no_list: list[int] = None, model_name: str = None):
    Path.mkdir(data_path / f'{year}/compounds', parents=True, exist_ok=True)
    if patent_no_list is None:
        pdf_files = list((data_path / str(year) / 'pages_with_markush').glob("*.pdf"))
    else:
        pdf_files = [data_path / str(year) / 'pages_with_markush' / f'{str(patent_no)}.pdf' for patent_no in patent_no_list]

    save_counter = 0
    fail_counter = 0
    extract_counter = 0
    pbar = tqdm(pdf_files, total=len(pdf_files), unit="Patent")
    pbar.set_postfix(save=save_counter, fail_rate="0.00%")
    for pdf_file in pbar:
        doc = Document(str(pdf_file))
        img_array_list = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=300, alpha=False)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()
            img_array_list.append(img_array)

        # Perform OCR on all pages in parallel
        workers = 64
        ocr_texts = [''] * len(img_array_list)
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(img_array_list), workers)) as executor:
            futures = {executor.submit(ocr_page, (i, p)): i for i, p in enumerate(img_array_list)}
            for future in concurrent.futures.as_completed(futures):
                page_index, text = future.result()
                ocr_texts[page_index] = text

        compounds = []
        for text in ocr_texts:
            iupac_names = extract_iupac_using_llm(text, model_name)
            if iupac_names != '':
                iupac_name_list = [name.strip() for name in iupac_names.split('\n') if name.strip()]
                compounds_one_page, fail_counter_one_page = iupac_to_smile(iupac_name_list)
                extract_counter += len(iupac_name_list)
                fail_counter += fail_counter_one_page
                compounds.extend(compounds_one_page)

        if len(compounds) != 0:
            df = pd.DataFrame(compounds, columns=['IUPAC', 'SMILES'])
            df.drop_duplicates(inplace=True)
            save_counter += 1
            pbar.set_postfix(save=save_counter, fail_rate=f"{fail_counter}/{extract_counter}={fail_counter/extract_counter:.2%}")
            df.to_csv(data_path / f'{year}/compounds/{pdf_file.stem}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Year of the patents")
    parser.add_argument("--patent_no_list", type=int, nargs='*', help="List of patent numbers")
    parser.add_argument("--model_name", type=str, default='gemma3:27b', help="Ollama model name")
    args = parser.parse_args()
    main(args.year, args.patent_no_list, args.model_name)