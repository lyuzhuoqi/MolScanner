from pathlib import Path
home = Path.home()
data_path = home / "projects/Markush/data"
from pymupdf import Document
from PIL import Image
import numpy as np
from paddleocr import PPStructureV3
pipeline = PPStructureV3(device='gpu', 
                         layout_detection_model_name='PP-DocLayout_plus-L',
                         text_recognition_model_name='en_PP-OCRv5_mobile_rec',
                         )

from pytesseract import image_to_string
import os
import concurrent 
import pandas as pd
import argparse
from tqdm import tqdm
import re


def ocr_page(page_data):
    """
    OCR function that works with ProcessPoolExecutor.
    Takes serialized image data.
    """
    page_index, img_array = page_data
    text = image_to_string(img_array)
    return page_index, text


def calculate_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def get_bbox_center(bbox):
    """get the center coordinate of a box"""
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return x_center, y_center


def is_bbox_overlap(bbox1, bbox2):
    """Identify whether two boxes are overlapped"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)


def is_text_covered_by_block(text_bbox, block_bboxes, page_width, overlap_threshold=0.5):
    """Identify whether a text box is covered by any other box"""
    for block_bbox in block_bboxes:
        x_min = block_bbox[0]
        y_min = block_bbox[1]
        x_max = block_bbox[2]
        y_max = block_bbox[3]
        if block_bbox[2] - block_bbox[0] > page_width/2: # for images across columns, extend their size
            x_min = 0
            x_max = page_width
        if is_bbox_overlap(text_bbox, (x_min, y_min, x_max, y_max)):
            x_overlap = max(0, min(text_bbox[2], x_max) - max(text_bbox[0], x_min))
            y_overlap = max(0, min(text_bbox[3], y_max) - max(text_bbox[1], y_min))
            overlap_area = x_overlap * y_overlap
            text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
            if text_area > 0 and overlap_area / text_area > overlap_threshold:
                return True
    return False


def calculate_y_distance_to_nearest_box(box, all_boxes, x_overlap_threshold=0.3):
    center_y = (box[1] + box[3]) / 2
    min_distance = float('inf')
    x0, x1 = box[0], box[2]
    width = x1 - x0

    for other_box in all_boxes:
        if other_box == box:
            continue
        ox0, ox1 = other_box[0], other_box[2]
        overlap = max(0, min(x1, ox1) - max(x0, ox0)) # calculate x overlap
        if width > 0 and overlap / width < x_overlap_threshold:
            continue  # x overlap is not enough to consider as same column
        other_center_y = (other_box[1] + other_box[3]) / 2
        y_distance = abs(center_y - other_center_y)
        min_distance = min(min_distance, y_distance)
    return min_distance if min_distance != float('inf') else 1000.0


def merge_and_sort_blocks(block_df, ocr_df, page_width, y_tolerance=15):
    """
    Combine image and text. Filter text covered by images.
    Order:
    1. Left column first, then right column
    2. Inside each column: sort by y ascending, if y is close, sort by x ascending
    """
    block_df = block_df[~block_df['block_label'].isin(['text', 'formula'])]
    block_bboxes = block_df['block_bbox'].tolist() # all bboxes for non-text

    # Filter text covered by images
    filtered_ocr = []
    for _, row in ocr_df.iterrows():
        text_bbox = row['box']
        if not is_text_covered_by_block(text_bbox, block_bboxes, page_width):
            filtered_ocr.append(row)
    if len(filtered_ocr) == 0:
        return None
    
    ocr_filtered_df = pd.DataFrame(filtered_ocr)

    # Combine text and image
    combined_df = pd.concat([
        block_df.loc[block_df.block_label=='image', ['block_label', 'block_content', 'block_bbox']].rename(columns={
            'block_label': 'type',
            'block_content': 'content',
            'block_bbox': 'box'
        }),
        ocr_filtered_df[['text', 'box', 'area', 'min_y_distance']].rename(columns={
            'text': 'content'
        }).assign(type='text')
    ], ignore_index=True)

    # Center of box
    centers = combined_df['box'].apply(get_bbox_center)
    combined_df['x_center'] = [c[0] for c in centers]
    combined_df['y_center'] = [c[1] for c in centers]

    # Mark column
    combined_df['column'] = combined_df['x_center'].apply(lambda x: 'left' if x < page_width / 2  else 'right')

    # Sort inside each column
    sorted_left = combined_df[combined_df['column'] == 'left'].copy()
    sorted_right = combined_df[combined_df['column'] == 'right'].copy()
    # Sorted by y, if y is close, sort by x
    sorted_left['y_group'] = (sorted_left['y_center'] // y_tolerance).astype(int)
    sorted_left = sorted_left.sort_values(by=['y_group', 'x_center']).drop(columns=['y_group'])
    sorted_right['y_group'] = (sorted_right['y_center'] // y_tolerance).astype(int)
    sorted_right = sorted_right.sort_values(by=['y_group', 'x_center']).drop(columns=['y_group'])

    result = pd.concat([sorted_left, sorted_right], ignore_index=True) # left column first, then right column

    final_df = result.drop(columns=['x_center', 'y_center', 'column']).reset_index(drop=True)
    return final_df


def remove_boxes_in_same_line(markush_boxes, next_box, page_width, y_tolerance=15):
    """
    Remove boxes from markush_boxes that are in the same column (left/right) and on the same line (y proximity) as next_box.

    Args:
        markush_boxes (list of tuples): List of (box, box_type), where box is [x0, y0, x1, y1]
        next_box (list or tuple): The target bounding box [x0, y0, x1, y1]
        page_width (int or float): Width of the page, used to determine left/right column
        y_tolerance (float): Tolerance in pixels for determining if two boxes are on the same line

    Returns:
        list: Filtered markush_boxes with overlapping boxes removed
    """
    # Calculate center of next_box
    nx0, ny0, nx1, ny1 = next_box
    next_center_x = (nx0 + nx1) / 2
    next_center_y = (ny0 + ny1) / 2

    # Determine which column the next_box belongs to
    next_column = 'left' if next_center_x < page_width / 2 else 'right'

    # Filter markush_boxes
    filtered_boxes = []
    for box, box_type in markush_boxes:
        bx0, by0, bx1, by1 = box
        b_center_x = (bx0 + bx1) / 2
        b_center_y = (by0 + by1) / 2

        # Determine column of current box
        b_column = 'left' if b_center_x < page_width / 2 else 'right'

        # Check if it is in the same column and vertically close (same line)
        if b_column == next_column and abs(b_center_y - next_center_y) < y_tolerance:
            continue  # Skip (remove this box)
        filtered_boxes.append((box, box_type))

    return filtered_boxes


def split_boxes_by_y_distance(group_boxes, threshold=10.0):
    """
    split a list of boxes into child lists based on y-distance
    
    Args:
        group_boxes: list of lists, every child list: [x_min, y_min, x_max, y_max]
        threshold: float, threshold for splitting

    Return:
        list of lists of boxes
    """
    if not group_boxes:
        return []
    
    result = []
    current_group = []

    for i, box in enumerate(group_boxes):
        x_min, y_min, x_max, y_max = box
        
        if i == 0:
            current_group.append(box) # add the start box
        else:
            prev_x_min, prev_y_min, prev_x_max, prev_y_max = group_boxes[i - 1]
            y_distance = y_min - prev_y_max  # the upper bound of the current box - the lower bound of the previous box

            # If y distance is greater than threshold, split
            if y_distance > threshold:
                result.append(current_group)
                current_group = []
            
            current_group.append(box)
    
    if current_group:
        result.append(current_group) # add the last group
    
    return result


def merge_boxes(img_array, markush_boxes, expand_margin, save_path):
    """
    Jointly display boxes for two column separately and merge them vertically

    Args:
        img_array: numpy array
        markush_boxes: list: [(box, type), ...]
        expand_margin: number of pixal to expand for all boarders
        save_path: Path to save. Don't save if None
    """
    img = Image.fromarray(img_array)
    
    # separate two columns
    left_boxes = []
    right_boxes = []
    for box, box_type in markush_boxes:
        x_center = (box[0] + box[2]) / 2
        if x_center < img.width / 2:
            left_boxes.append(box)
        else:
            right_boxes.append(box)
    
    left_regions = []
    right_regions = []
    
    # left
    if left_boxes:
        groups  = split_boxes_by_y_distance(left_boxes, threshold=95)
        for group_boxes in groups:
            x_mins = [box[0] for box in group_boxes]
            y_mins = [box[1] for box in group_boxes]
            x_maxs = [box[2] for box in group_boxes]
            y_maxs = [box[3] for box in group_boxes]

            left_x_min = img.width/8
            left_y_min = max(0, min(y_mins) - expand_margin)
            left_x_max = max(img.width/2, max(x_maxs) + expand_margin)
            left_y_max = min(img.height, max(y_maxs) + expand_margin)
            
            left_region = img.crop((left_x_min, left_y_min, left_x_max, left_y_max))
            left_regions.append(left_region)
    
    # right
    if right_boxes:
        groups  = split_boxes_by_y_distance(right_boxes, threshold=95)
        for group_boxes in groups:
            x_mins = [box[0] for box in group_boxes]
            y_mins = [box[1] for box in group_boxes]
            x_maxs = [box[2] for box in group_boxes]
            y_maxs = [box[3] for box in group_boxes]

            right_x_min = min(img.width/2, min(x_mins) - expand_margin)
            right_y_min = max(0, min(y_mins) - expand_margin)
            right_x_max = img.width*7/8
            right_y_max = min(img.height, max(y_maxs) + expand_margin)
            
            right_region = img.crop((right_x_min, right_y_min, right_x_max, right_y_max))
            right_regions.append(right_region)

    # combine all regions (left first then right, vertically)
    all_regions = left_regions + right_regions
    
    if len(all_regions) > 0:
        if len(all_regions) == 1:
            final_image = all_regions[0]
        else:
            # combine vertically
            total_width = max(region.width for region in all_regions)
            total_height = sum(region.height for region in all_regions)
            
            final_image = Image.new('RGB', (total_width, total_height), color='white')
            
            current_y = 0
            for region in all_regions:
                final_image.paste(region, (0, current_y))
                current_y += region.height
        
        if save_path:
            final_image.save(save_path)


def main(year: int, patent_no_list: list[int] = None):
    Path.mkdir(data_path / f'{year}/claimed_markush', parents=True, exist_ok=True)
    if patent_no_list is None:
        pdf_files = list((data_path / str(year) / 'pages_with_markush').glob("*.pdf"))
    else:
        pdf_files = [data_path / str(year) / 'pages_with_markush' / f'{str(patent_no)}.pdf' for patent_no in patent_no_list]

    save_counter = 0
    no_claim_counter = 0
    no_claim_sent_counter = 0
    no_img_claimed = 0
    img_wo_R_counter = 0
    cross_page_counter = 0
    peptide_counter = 0

    pbar = tqdm(pdf_files, total=len(pdf_files), unit="Patent")
    pbar.set_postfix(save=save_counter, no_claim=no_claim_counter, no_img_claimed=no_img_claimed, 
                     no_claim_sent=no_claim_sent_counter, img_wo_R=img_wo_R_counter, cross_page=cross_page_counter, peptide=peptide_counter)
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

        claim_page_index = [i for i, text in enumerate(ocr_texts) if 'claim' in text]
        img_array_list = [img_array_list[i] for i in claim_page_index]

        if len(img_array_list) == 0:
            no_claim_counter += 1
            pbar.set_postfix(save=save_counter, no_claim=no_claim_counter, no_img_claimed=no_img_claimed, 
                             no_claim_sent=no_claim_sent_counter, img_wo_R=img_wo_R_counter,
                             cross_page=cross_page_counter, peptide=peptide_counter)
            continue

        structure_output = pipeline.predict(img_array_list, 
                                            use_seal_recognition=False, 
                                            # use_formula_recognition=False
                                        )

        claim_sents = ["What is claimed is",
                       "The invention claimed is",
                       "I claim",
                       "We claim"]
        peptide_key_words = ['polypeptide', 'amino acid sequence']
        NUMBERED_LIST_PATTERN = r'^\d+\.\s'

        valid_page_index = None
        for page_index, (output, img_array) in enumerate(zip(structure_output, img_array_list)):
            page_width = img_array.shape[1]
            void_min_1, void_max_1 = page_width * 0.45, page_width * 0.50 # right side of the left column
            void_min_2, void_max_2 = page_width * 0.80, page_width * 0.90 # right side of the right column

            # Prepare block_df, which contains image boxes on the page
            block_df = pd.DataFrame(output.json['res']['parsing_res_list'])

            # Prepare ocr_df, which contains text boxes on the page
            ocr_df = pd.DataFrame(columns=['text', 'box'])
            ocr_df['text'] = output.json['res']['overall_ocr_res']['rec_texts']
            ocr_df['box'] = output.json['res']['overall_ocr_res']['rec_boxes']
            ocr_df.drop(ocr_df[(ocr_df['box'].apply(lambda box: 275 < box[1] and box[3] < 360))].index, inplace=True) # drop page number
            ocr_df.drop(ocr_df[(ocr_df['box'].apply(lambda box: void_min_1 < (box[0]+box[2])/2 < void_max_1))].index, inplace=True) # drop line number
            # ocr_df.drop(ocr_df[(ocr_df['box'].apply(lambda box: void_min_2 < (box[0]+box[2])/2 < void_max_2))].index, inplace=True) # drop equation number
            ocr_df['area'] = ocr_df['box'].apply(calculate_box_area)
            all_boxes = ocr_df['box'].tolist()
            ocr_df['min_y_distance'] = ocr_df['box'].apply(lambda box: calculate_y_distance_to_nearest_box(box, all_boxes))
            ocr_df = ocr_df[~((ocr_df['area'] < 2000) & (ocr_df['min_y_distance'] > 80))] # drop isolated small text boxes

            sorted_df = merge_and_sort_blocks(block_df, ocr_df, page_width=page_width)  # combine block_df and ocr_df

            if sorted_df is None:
                continue

            if not any(sent in ''.join(sorted_df.content) for sent in claim_sents):
                # No claim section found
                continue

            sorted_df.loc[:, 'box'] = sorted_df['box'].apply(
                lambda box_list: [box_list[0]-10, box_list[1], box_list[2], box_list[3]+5]
            )

            img = Image.fromarray(img_array)
            markush_boxes = []
            claim_row_index = None
            is_in_claim_section = False
            for row_index, row in sorted_df.iterrows():
                if any(sent in row.content for sent in claim_sents) and not is_in_claim_section:
                    is_in_claim_section = True
                    claim_row_index = row_index
                    break
            if not is_in_claim_section:
                # No claim section found
                continue
            
            candidate_df = sorted_df[sorted_df.index > claim_row_index]
            if len(candidate_df[candidate_df['type']=='image']) == 0:
                no_img_claimed += 1
                pbar.set_postfix(save=save_counter, no_claim=no_claim_counter, no_img_claimed=no_img_claimed, 
                                 no_claim_sent=no_claim_sent_counter, img_wo_R=img_wo_R_counter, 
                                 cross_page=cross_page_counter, peptide=peptide_counter)
                break

            first_img_in_claim = candidate_df[candidate_df['type']=='image'].iloc[0]
            if any(key_word in ''.join(sorted_df[sorted_df.index < first_img_in_claim.name]['content']).lower() for key_word in peptide_key_words):
                peptide_counter += 1
                pbar.set_postfix(save=save_counter, no_claim=no_claim_counter,no_img_claimed=no_img_claimed, 
                                 no_claim_sent=no_claim_sent_counter, img_wo_R=img_wo_R_counter,cross_page=cross_page_counter, peptide=peptide_counter)
                break
            img_ocr_content = image_to_string(img.copy().crop(tuple(first_img_in_claim.box)), config='--psm 6')
            if 'R' in img_ocr_content or 'R' in first_img_in_claim.content:
                first_claim_img_row_index = first_img_in_claim.name
                current_box = first_img_in_claim.box
                markush_boxes.append((current_box, 'image'))  # save boxes and their types
            else:
                img_wo_R_counter += 1
                pbar.set_postfix(save=save_counter, no_claim=no_claim_counter, no_img_claimed=no_img_claimed, 
                                 no_claim_sent=no_claim_sent_counter, img_wo_R=img_wo_R_counter,
                                 cross_page=cross_page_counter, peptide=peptide_counter)
                break

            valid_termination = False
            current_row_index = first_claim_img_row_index
            while not valid_termination:
                candidate_df = sorted_df[sorted_df.index > current_row_index]
                if len(candidate_df) == 0:
                    cross_page_counter += 1
                    pbar.set_postfix(save=save_counter, no_claim=no_claim_counter, 
                                     no_img_claimed=no_img_claimed, no_claim_sent=no_claim_sent_counter, img_wo_R=img_wo_R_counter, cross_page=cross_page_counter, 
                                     peptide=peptide_counter)
                    break

                next_row = candidate_df.iloc[0]
                next_row_index = next_row.name
                ocr_result = image_to_string(img.copy().crop(tuple(next_row.box)), config='--psm 6')

                if next_row.type == 'image': # if reach a image box:
                    markush_boxes.append((next_row.box, 'image'))
                    current_row_index = next_row_index
                    continue

                if next_row.type == 'text': # if reach a text box:
                    # check whether the text box starts a new claim
                    # if ocr_result.strip().startswith('2. ') and next_row.content.strip().startswith('2. '): 
                    if re.match(NUMBERED_LIST_PATTERN, ocr_result.strip()) or re.match(NUMBERED_LIST_PATTERN, next_row.content.strip()):
                        markush_boxes = remove_boxes_in_same_line(
                            markush_boxes=markush_boxes,
                            next_box=next_row.box,
                            page_width=page_width,
                            y_tolerance=15
                        )
                        valid_termination = True
                        break

                    markush_boxes.append((next_row.box, 'text')) # don't include the next claim
                    current_row_index = next_row_index
                    continue

            if not is_in_claim_section:
                no_claim_sent_counter += 1
                pbar.set_postfix(save=save_counter, no_claim=no_claim_counter, no_img_claimed=no_img_claimed, 
                                 no_claim_sent=no_claim_sent_counter, img_wo_R=img_wo_R_counter, 
                                 cross_page=cross_page_counter, peptide=peptide_counter)
            if valid_termination and any(box_type == 'text' for _, box_type in markush_boxes):
                markush_boxes = [(sorted_df.iloc[i].box, 'text') 
                                 for i in range(claim_row_index, first_claim_img_row_index+1)] + markush_boxes
                valid_page_index = page_index
                save_counter += 1
                pbar.set_postfix(save=save_counter, no_claim=no_claim_counter, no_img_claimed=no_img_claimed, 
                                 no_claim_sent=no_claim_sent_counter, img_wo_R=img_wo_R_counter, cross_page=cross_page_counter, peptide=peptide_counter)
                merge_boxes(img_array_list[valid_page_index], 
                            markush_boxes, 
                            expand_margin=0,
                            save_path=data_path / f'{year}/claimed_markush' / f'{pdf_file.stem}.png')
                break
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract claim sections from patent PDFs which contains Markush structure.")
    parser.add_argument("--year", type=int, required=True, help="Year of the patents to process.")
    parser.add_argument("--patent_no_list", type=int, nargs='*', default=None, help="Specific patent numbers to process. If not provided, all patents in the specified year will be processed.")
    args = parser.parse_args()
    main(args.year, args.patent_no_list)