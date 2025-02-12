from doctr.io import DocumentFile
from doctr.models import kie_predictor
import json
import numpy as np
import re
from typing import List, Tuple, Set
from itertools import chain, accumulate
import fitz  # PyMuPDF
from PIL import Image
import io
import os

def ocrtotext(filepath):
    """Convert PDF to text using OCR."""
    # Model
    model = kie_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    # PDF
    doc = DocumentFile.from_pdf(filepath)
    # Analyze
    result = model(doc)
    
    pagelist = []
    for i in range(len(result.pages)):
        predlist = []
        predictions = result.pages[i].predictions
        for class_name in predictions.keys():
            for prediction in predictions[class_name]:
                predlist.append(prediction.mrvalues()) 
        pagelist.append(predlist)
    
    return pagelist

def clean_filename(number: str) -> str:
    """Clean the number to contain only digits and convert forward slashes to percent signs.
    Examples:
        "31/1" -> "31%1"
        "31/1-" -> "31%1"
        "4/2am" -> "4%2"
    """
    # First remove everything that's not a digit or forward slash
    cleaned = re.sub(r'[^0-9/]', '', number)
    # Then replace remaining forward slashes with percent signs
    return cleaned.replace('/', '%')

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    directories = ['pictures', 'pictures/duplicates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def save_image_with_duplicate_handling(filename, output_pdf, x1, y1, x2, y2, page_number=0):
    """Save image with simple duplicate handling."""
    ensure_directories_exist()
    
    # Check if file already exists in pictures directory
    if os.path.exists(output_pdf):
        # If it exists, save to duplicates folder with same name
        base_name = os.path.basename(output_pdf)
        duplicate_path = os.path.join('pictures/duplicates', base_name)
        print(f"Duplicate found. Saving to: {duplicate_path}")
        crop_pdf_to_image(filename, duplicate_path, x1, y1, x2, y2, page_number=page_number)
    else:
        # If it doesn't exist, save to original pictures directory
        crop_pdf_to_image(filename, output_pdf, x1, y1, x2, y2, page_number=page_number)

def crop_pdf_to_image(input_path, output_path, x1, y1, x2, y2, page_number=0, dpi=300):
    """Crop a section of a PDF page and save it as an image."""
    try:
        # Open the PDF
        doc = fitz.open(input_path)
        page = doc[page_number]
        
        # Get the page dimensions
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Convert decimal coordinates to actual points
        actual_x1 = x1 * page_width
        actual_y1 = y1 * page_height
        actual_x2 = x2 * page_width
        actual_y2 = y2 * page_height
        
        # Create the crop rectangle
        crop_rect = fitz.Rect(actual_x1, actual_y1, actual_x2, actual_y2)
        
        # Get the cropped pixmap (image)
        # Set matrix for desired DPI
        zoom = dpi / 72  # 72 is the default PDF DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        # Get pixmap of the cropped area
        pix = page.get_pixmap(matrix=matrix, clip=crop_rect)
        
        # Save the image
        pix.save(output_path)
        
        doc.close()
        
        print(f"Successfully saved cropped image to {output_path}")
        print(f"Converted coordinates: ({actual_x1}, {actual_y1}) to ({actual_x2}, {actual_y2})")
        print(f"Image dimensions: {pix.width}x{pix.height} pixels")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def clean_text(text):
    """Clean up text by removing special characters and normalizing spaces"""
    text = text.replace('-', ' ')  # Replace hyphens with spaces
    text = re.sub(r'[,.]$', '', text)  # Remove trailing comma or period
    return ' '.join(text.split())

def is_number(text):
    """Check if text contains a valid poll number."""
    if not text:
        return False
    
    # Remove everything except digits and forward slashes
    cleaned = re.sub(r'[^0-9/]', '', text)
    
    if not cleaned:
        return False
        
    # Split into parts if there's a forward slash
    parts = cleaned.split('/')
    main_number = parts[0]
    
    # Check if main number is valid (no leading zeros unless it's just "0")
    if not main_number or (len(main_number) > 1 and main_number.startswith('0')):
        return False
    
    # If there's a sub-number, check if it's valid
    if len(parts) > 1:
        sub_number = parts[1]
        if not sub_number or (len(sub_number) > 1 and sub_number.startswith('0')):
            return False
    
    return True

def should_exclude_group(texts):
    """Check if a group of texts forms an excluded phrase."""
    # Join texts and convert to lowercase for comparison
    text_group = ' '.join(texts).lower()
    
    # List of phrases to exclude
    exclude_phrases = [
        'page'
    ]
    
    return any(phrase in text_group for phrase in exclude_phrases)

def get_bounding_box(coordinates_list):
    """Calculate bounding box from a list of coordinates."""
    if not coordinates_list:
        return None
    
    x_coords = []
    y_coords = []
    for (x1, y1), (x2, y2) in coordinates_list:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def group_entries_by_column(line_entries, page_width=1.0):
    """Split entries into left and right columns based on x-coordinate."""
    left_column = []
    right_column = []
    
    # Use middle of page as split point
    middle_x = page_width / 2
    
    for entry in line_entries:
        text, coords = entry
        x1 = coords[0][0]  # Get x-coordinate of start point
        
        if x1 < middle_x:
            left_column.append(entry)
        else:
            right_column.append(entry)
    
    return left_column, right_column

def group_entries_by_position(line_entries, x_threshold=0.1, page_width=1.0):
    """Group entries based on their x-coordinate proximity, handling double columns."""
    if not line_entries:
        return []
    
    # Split entries into columns
    left_entries, right_entries = group_entries_by_column(line_entries, page_width)
    
    groups = []
    
    # Process each column separately
    for column_entries in [left_entries, right_entries]:
        if not column_entries:
            continue
            
        current_group = []
        # Sort by x-coordinate within the column
        sorted_entries = sorted(column_entries, key=lambda x: x[1][0][0])
        
        last_x = None
        for text, coords in sorted_entries:
            current_x = coords[0][0]
            
            if last_x is None or current_x - last_x < x_threshold:
                current_group.append((text, coords))
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [(text, coords)]
            
            last_x = current_x
        
        if current_group:
            groups.append(current_group)
    
    return groups

def extract_entries(poll_data):
    """Extract entries from poll data, handling double columns."""
    results = []
    data = eval(poll_data)
    
    # Sort by y-coordinate first
    data.sort(key=lambda x: x[1][0][1])
    
    # Group by similar y-coordinates
    y_threshold = 0.01
    current_y = None
    current_line = []
    lines = []
    
    for entry in data:
        text, ((_, y1), _) = entry
        
        if current_y is None:
            current_y = y1
            current_line.append(entry)
        elif abs(y1 - current_y) < y_threshold:
            current_line.append(entry)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [entry]
            current_y = y1
    
    if current_line:
        lines.append(current_line)
    
    # Process each line
    for line in lines:
        entry_groups = group_entries_by_position(line)
        
        for group in entry_groups:
            texts = [entry[0] for entry in group]
            
            # Skip groups that form excluded phrases
            if should_exclude_group(texts):
                continue
            
            # Find the number (if any) in this group
            number = None
            number_coords = None
            name_parts = []
            name_coords = []
            
            # First find the number and its position
            number_x = float('inf')  # Initialize to a large value
            for text, coords in group:
                if is_number(text):
                    number = text
                    number_coords = coords
                    number_x = coords[0][0]  # x-coordinate of number's start
            
            # Then collect name parts that are to the right of the number
            rightmost_name_x = float('-inf')  # Initialize to a small value
            for text, coords in group:
                if not is_number(text):
                    name_x = coords[0][0]  # x-coordinate of name part's start
                    name_parts.append(text)
                    name_coords.append(coords)
                    rightmost_name_x = max(rightmost_name_x, name_x)
            
            # Only process groups that have a number and name parts
            # AND where the number appears to the left of the name
            if number and name_parts and number_x < rightmost_name_x:
                name = clean_text(' '.join(name_parts))
                if name:
                    all_coords = [number_coords] + name_coords
                    bbox = get_bounding_box(all_coords)
                    if bbox:
                        results.append((number, name, bbox))
    
    return results

def format_results(entries):
    """Format the results in a clean, readable format"""
    formatted = []
    for num, name, bbox in entries:
        formatted.append([num, name, bbox])
    return formatted

def parse_entry_number(entry_num: str) -> Tuple[float, float]:
    """Parse entry numbers into sortable tuples."""
    # Clean the number to only contain digits and forward slashes
    cleaned = re.sub(r'[^0-9/]', '', entry_num)
    
    if '/' in cleaned:
        main, sub = cleaned.split('/')
        return (float(main), float(sub))
    return (float(cleaned), 0)

def get_main_number(entry_num: str) -> int:
    """Extract the main number from an entry."""
    # Clean the number to only contain digits and forward slashes
    cleaned = re.sub(r'[^0-9/]', '', entry_num)
    return int(cleaned.split('/')[0])

def find_missing_numbers(entries: List[List]) -> Set[int]:
    """Find missing main numbers in the sequence."""
    # Get all main numbers
    main_numbers = {get_main_number(entry[0]) for entry in entries}
    
    # Find the range
    min_num = min(main_numbers)
    max_num = max(main_numbers)
    
    # Create set of all numbers that should exist
    expected_numbers = set(range(min_num, max_num + 1))
    
    # Find missing numbers
    return expected_numbers - main_numbers

def sort_and_check_entries(entries: List[List]) -> List[List]:
    """Sort entries based on their numbers."""
    # Sort based on entry numbers
    sorted_entries = sorted(entries, key=lambda x: parse_entry_number(x[0]))
    return sorted_entries

def get_main_number_safe(number_str: str) -> int:
    """Safely extract the main number from an entry, handling potential OCR errors."""
    try:
        # Remove everything except digits and slashes
        cleaned = re.sub(r'[^0-9/]', '', number_str)
        if not cleaned:
            return 0
        # Take the part before any slash
        main_part = cleaned.split('/')[0]
        return int(main_part)
    except:
        return 0

def analyze_page_numbers(entries) -> Tuple[int, int]:
    """Analyze numbers on a page to determine reasonable range."""
    numbers = []
    for entry in entries:
        number = entry[0]  # Get the number part
        main_num = get_main_number_safe(number)
        if main_num > 0:  # Ignore 0s from failed conversions
            numbers.append(main_num)
    
    if not numbers:
        return (0, float('inf'))  # No numbers to analyze
        
    # Get basic statistics
    median = sorted(numbers)[len(numbers)//2]
    
    # Set reasonable limits
    # Upper limit is either 2x the median or 120, whichever is larger
    # This handles both dense and sparse pages
    upper_limit = max(median * 2, 120)
    
    return (0, upper_limit)

def is_reasonable_number(number: str, page_limits: Tuple[int, int]) -> bool:
    """Check if a number is within reasonable limits for the page."""
    main_num = get_main_number_safe(number)
    min_limit, max_limit = page_limits
    
    # Check if the number is within reasonable limits
    return min_limit <= main_num <= max_limit

def process_poll_data(poll_data):
    """Extract and format number-name pairs from poll data with bounding boxes"""
    # First do a preliminary extraction
    preliminary_entries = extract_entries(poll_data)
    
    # Analyze the page to determine reasonable number limits
    page_limits = analyze_page_numbers(preliminary_entries)
    print(f"Page limits determined: up to {page_limits[1]}")
    
    # Filter entries based on reasonable limits
    filtered_entries = [
        entry for entry in preliminary_entries
        if is_reasonable_number(entry[0], page_limits)
    ]
    
    return format_results(filtered_entries)

def complete_proccess(filename):
    """Main process to extract and save all entries from a PDF."""
    # Get OCR results
    raw_text = ocrtotext(filename)
    print(f"Number of pages processed: {len(raw_text)}")
    
    # Open PDF to get page dimensions
    doc = fitz.open(filename)
    # Process each page and keep track of page numbers
    result = []
    page_entries = []
    
    for page_num, page_data in enumerate(raw_text):
        # Process the page data with page dimensions
        page_results = process_poll_data(str(page_data))
        
        # Add page number to each entry
        for entry in page_results:
            page_entries.append((entry, page_num))
            result.append(entry)
    doc.close()
    
    # Sort entries
    sorted_entries = sort_and_check_entries(result)
    
    # Create a mapping of entries to their page numbers
    entry_to_page = {tuple(entry[0:2]): page_num for entry, page_num in page_entries}
    
    # Process regular entries
    for entry in sorted_entries:
        number, name, (x1, y1, x2, y2) = entry
        page_num = entry_to_page.get(tuple(entry[0:2]), 0)
        
        # Clean the number only when creating the filename
        filename_number = re.sub(r'[^0-9/]', '', number)  # Remove everything except digits and /
        filename_number = filename_number.replace('/', '%')  # Replace / with %
        output_pdf = f"pictures/{filename_number}.png"
        save_image_with_duplicate_handling(filename, output_pdf, x1, y1, x2, y2, page_number=page_num)

# Run the process
if __name__ == "__main__":
    complete_proccess("Testing/Hil-a.pdf")