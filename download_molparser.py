import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def main():
    save_dir = "/home/zqlyu2/projects/Markush/data/molparser_sft_real"
    image_dir = os.path.join(save_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    print("Loading MolParser-7M dataset (sft_real split)...")
    # This will download the dataset from HuggingFace
    ds = load_dataset("UniParser/MolParser-7M", name="sft_real", split="train")
    
    records = []
    print(f"Total samples to process: {len(ds)}")
    
    for i, item in enumerate(tqdm(ds)):
        image_id = str(i)
        img = item["image"]
        smiles = item["SMILES"]
        
        # Ensure image is in RGB format (sometimes mode might be L or RGBA)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img_path = os.path.join(image_dir, f"{image_id}.png")
        img.save(img_path)
        
        # Use relative path consistent with other data conventions
        rel_path = f"molparser_sft_real/images/{image_id}.png"
        records.append({
            "image_id": image_id,
            "file_path": rel_path,
            "SMILES": smiles
        })
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, "sft_real.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSuccessfully saved {len(df)} records to {csv_path}")

if __name__ == "__main__":
    main()
