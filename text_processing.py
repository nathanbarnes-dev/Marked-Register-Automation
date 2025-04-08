from doctr.io import DocumentFile
from doctr.models import kie_predictor
import re
from typing import List, Tuple, Set
from dataclasses import dataclass
from typing import Optional

@dataclass
class Entry:
    """Data class to represent a poll entry."""
    number: str
    name: str
    bbox: Tuple[float, float, float, float]
    page: Optional[int] = None

def ocrtotext(filepath: str) -> List[List]:
    """Convert PDF to text using OCR."""
    model = kie_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    doc = DocumentFile.from_pdf(filepath)
    for i, page in enumerate(doc):
        result = model([page])  # Process one page at a time
        print(f"Finished processing page {i + 1}")
    result = model(doc)
    
    return [
        [prediction.mrvalues() for class_name in page.predictions 
         for prediction in page.predictions[class_name]]
        for page in result.pages
    ]

def clean_text(text: str) -> str:
    """Clean up text by removing special characters and normalizing spaces."""
    text = text.replace('-', ' ')
    text = re.sub(r'[,.]$', '', text)
    return ' '.join(text.split())

def is_number(text: str) -> bool:
    """Check if text contains a valid poll number."""
    if not text:
        return False
    
    cleaned = re.sub(r'[^0-9/]', '', text)
    if not cleaned:
        return False
        
    parts = cleaned.split('/')
    main_number = parts[0]
    
    if not main_number or (len(main_number) > 1 and main_number.startswith('0')):
        return False
    
    if len(parts) > 1:
        sub_number = parts[1]
        if not sub_number or (len(sub_number) > 1 and sub_number.startswith('0')):
            return False
    
    return True

def should_exclude_group(texts: List[str]) -> bool:
    """Check if a group of texts forms an excluded phrase."""
    text_group = ' '.join(texts).lower()
    exclude_phrases = ['page']
    return any(phrase in text_group for phrase in exclude_phrases)

def get_bounding_box(coordinates_list: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Optional[Tuple[float, float, float, float]]:
    """Calculate bounding box from a list of coordinates."""
    if not coordinates_list:
        return None
    
    x_coords = []
    y_coords = []
    for (x1, y1), (x2, y2) in coordinates_list:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def group_entries_by_column(line_entries: List[Tuple[str, Tuple]], page_width: float = 1.0) -> Tuple[List, List]:
    """Split entries into left and right columns based on x-coordinate."""
    middle_x = page_width / 2
    left_column = []
    right_column = []
    
    for entry in line_entries:
        text, coords = entry
        x1 = coords[0][0]
        
        if x1 < middle_x:
            left_column.append(entry)
        else:
            right_column.append(entry)
    
    return left_column, right_column

def group_entries_by_position(line_entries: List[Tuple[str, Tuple]], x_threshold: float = 0.1, page_width: float = 1.0) -> List[List]:
    """Group entries based on their x-coordinate proximity, handling double columns."""
    if not line_entries:
        return []
    
    left_entries, right_entries = group_entries_by_column(line_entries, page_width)
    groups = []
    
    for column_entries in [left_entries, right_entries]:
        if not column_entries:
            continue
            
        current_group = []
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


def extract_entries(poll_data: str) -> List[Entry]:
    """Combined approach that directly combines results from both methods.
    
    This implementation simply runs both extraction methods and combines
    their results without trying to normalize or deduplicate, ensuring
    we don't lose any entries in the process.
    """
    # Define the two extraction methods
    def extract_entries_v1(poll_data: str) -> List[Entry]:
        """Extract entries with simpler line-by-line processing."""
        results = []
        data = eval(poll_data)
        
        # Sort by y-coordinate first to group entries by line
        data.sort(key=lambda x: x[1][0][1])
        
        # Use adaptive thresholding for line grouping
        lines = []
        current_line = []
        
        for i, entry in enumerate(data):
            text, coords = entry
            
            if i == 0:
                # First entry starts a new line
                current_line = [entry]
            else:
                prev_y = data[i-1][1][0][1]
                curr_y = coords[0][1]
                
                # Compute distance between this entry and previous one
                y_diff = abs(curr_y - prev_y)
                
                # Use an adaptive threshold based on font size approximation
                font_height = coords[1][1] - coords[0][1]
                y_threshold = min(0.01, font_height * 0.5)  # Adaptive threshold
                
                if y_diff < y_threshold:
                    # Same line
                    current_line.append(entry)
                else:
                    # New line
                    if current_line:
                        lines.append(current_line)
                    current_line = [entry]
        
        if current_line:
            lines.append(current_line)
        
        # Process each line
        for line in lines:
            # Sort entries in line by x-coordinate
            line.sort(key=lambda x: x[1][0][0])
            
            # Check if there are poll numbers in this line
            has_poll_number = any(is_number(entry[0]) for entry in line)
            
            # If no poll numbers, this might be a header or non-entry line
            if not has_poll_number:
                continue
            
            # Process each entry in the line that has a poll number
            for i, (text, coords) in enumerate(line):
                if is_number(text):
                    # Found a poll number
                    poll_num = text
                    poll_coords = coords
                    
                    # Collect all text after the poll number for name
                    name_parts = []
                    name_coords = []
                    
                    # Look at the entries after this one in the same line
                    for j in range(i + 1, len(line)):
                        next_text, next_coords = line[j]
                        
                        # If we encounter another poll number, that's a new entry
                        if is_number(next_text):
                            break
                        
                        name_parts.append(next_text)
                        name_coords.append(next_coords)
                    
                    # Create entry if we have both poll number and name
                    if name_parts:
                        name = clean_text(' '.join(name_parts))
                        if name:
                            all_coords = [poll_coords] + name_coords
                            bbox = get_bounding_box(all_coords)
                            if bbox:
                                results.append(Entry(number=poll_num, name=name, bbox=bbox))
        
        return results

    def extract_entries_v2(poll_data: str) -> List[Entry]:
        """Extract entries specifically for electoral register format."""
        results = []
        data = eval(poll_data)
        
        # Determine page dimensions from the data
        x_coords = [coord[0][0] for _, coord in data]
        y_coords = [coord[0][1] for _, coord in data]
        page_width = max(x_coords) - min(x_coords)
        
        # Identify middle point to separate left and right columns
        middle_x = (min(x_coords) + max(x_coords)) / 2
        
        # Group by rows based on y-coordinate
        data_by_y = sorted(data, key=lambda x: x[1][0][1])
        rows = []
        current_row = []
        
        for i, entry in enumerate(data_by_y):
            text, coords = entry
            
            if i == 0:
                current_row = [entry]
            else:
                prev_y = data_by_y[i-1][1][0][1]
                curr_y = coords[0][1]
                
                # Use a threshold appropriate for your document
                y_diff = abs(curr_y - prev_y)
                threshold = 0.015  # Adjust based on your document spacing
                
                if y_diff < threshold:
                    # Same row
                    current_row.append(entry)
                else:
                    # New row
                    if current_row:
                        rows.append(current_row)
                    current_row = [entry]
        
        if current_row:
            rows.append(current_row)
        
        # Process each row to extract entries from both columns
        for row in rows:
            # Split row into left and right columns
            left_column = [entry for entry in row if entry[1][0][0] < middle_x]
            right_column = [entry for entry in row if entry[1][0][0] >= middle_x]
            
            # Function to process a column and extract entries
            def process_column(column_entries):
                if not column_entries:
                    return
                    
                # Sort by x position
                column_sorted = sorted(column_entries, key=lambda x: x[1][0][0])
                
                # Look for poll number pattern
                potential_poll_numbers = []
                for i, (text, coords) in enumerate(column_sorted):
                    # Check if this is a likely poll number (including those with slashes)
                    if re.match(r'^\d+$', text) or re.match(r'^\d+[A-Za-z]$', text) or re.match(r'^\d+/\d+$', text):
                        potential_poll_numbers.append((i, text, coords))
                
                # Process each potential poll number
                for poll_idx, poll_text, poll_coords in potential_poll_numbers:
                    # Initialize for this poll entry
                    name_parts = []
                    name_coords = []
                    
                    # Check if there's a single letter after the poll number
                    letter_idx = poll_idx + 1
                    has_letter = False
                    
                    if letter_idx < len(column_sorted):
                        letter_text, letter_coords = column_sorted[letter_idx]
                        if re.match(r'^[A-Za-z]$', letter_text):
                            has_letter = True
                            letter_idx += 1  # Skip the letter in name collection
                    
                    # Collect name parts until next poll number or end of column
                    for j in range(poll_idx + 1 if not has_letter else letter_idx, len(column_sorted)):
                        next_text, next_coords = column_sorted[j]
                        
                        # Skip dashes and page numbers
                        if next_text == "-----" or next_text == "----" or re.match(r'^\d+$', next_text):
                            continue
                        
                        # Stop if this looks like another poll number
                        if re.match(r'^\d+$', next_text) or re.match(r'^\d+[A-Za-z]$', next_text) or re.match(r'^\d+/\d+$', next_text):
                            break
                        
                        # Add to name parts
                        name_parts.append(next_text)
                        name_coords.append(next_coords)
                    
                    # Create entry if we have name parts
                    if name_parts:
                        name = clean_text(' '.join(name_parts))
                        if name:
                            all_coords = [poll_coords] + name_coords
                            bbox = get_bounding_box(all_coords)
                            if bbox:
                                # Include the letter in the poll number if present
                                if has_letter:
                                    letter = column_sorted[poll_idx + 1][0]
                                    poll_num = f"{poll_text} {letter}"
                                else:
                                    poll_num = poll_text
                                    
                                results.append(Entry(number=poll_num, name=name, bbox=bbox))
            
            # Process both columns
            process_column(left_column)
            process_column(right_column)
        
        return results
    
    # Run both extraction methods
    results_v1 = extract_entries_v1(poll_data)
    results_v2 = extract_entries_v2(poll_data)
    
    # Simply combine the results from both methods
    combined_results = results_v1 + results_v2
    
    # Create a dictionary that maps poll numbers to entries
    # This serves as a simple way to deduplicate while preserving entries
    result_dict = {}
    
    # Add all entries to the dictionary
    # If there's a duplicate, keep both to be safe (using a unique key)
    for entry in combined_results:
        # Try to standardize the poll number format slightly
        clean_number = entry.number.strip()
        
        # If the entry already exists, add a counter to make it unique
        base_key = clean_number
        key = base_key
        counter = 1
        
        while key in result_dict:
            key = f"{base_key}_{counter}"
            counter += 1
        
        result_dict[key] = entry
    
    # Get all entries as a list
    results = list(result_dict.values())
    
    # Sort the results by poll number
    results.sort(key=lambda x: parse_entry_number(x.number))
    
    # For debugging: log the number of entries from each source
    with open("extraction_results.txt", "w") as f:
        f.write(f"Version 1 found {len(results_v1)} entries\n")
        f.write(f"Version 2 found {len(results_v2)} entries\n")
        f.write(f"Combined approach has {len(results)} entries\n\n")
        
        # Log all poll numbers
        f.write("Poll numbers from Version 1:\n")
        for entry in results_v1:
            f.write(f"{entry.number}\n")
        
        f.write("\nPoll numbers from Version 2:\n")
        for entry in results_v2:
            f.write(f"{entry.number}\n")
        
        f.write("\nPoll numbers in combined result:\n")
        for entry in results:
            f.write(f"{entry.number}\n")
    
    return results

def get_main_number_safe(number_str: str) -> int:
    """Safely extract the main number from an entry."""
    try:
        cleaned = re.sub(r'[^0-9/]', '', number_str)
        return int(cleaned.split('/')[0]) if cleaned else 0
    except:
        return 0

def analyze_page_numbers(entries: List[Entry]) -> Tuple[int, int]:
    """Analyze numbers on a page to determine reasonable range."""
    numbers = [get_main_number_safe(entry.number) for entry in entries]
    numbers = [n for n in numbers if n > 0]
    
    if not numbers:
        return (0, float('inf'))
    
    median = sorted(numbers)[len(numbers)//2]
    upper_limit = max(median * 2, 120)
    
    return (0, upper_limit)

def parse_entry_number(entry_num: str) -> Tuple[float, float]:
    """Parse entry numbers into sortable tuples."""
    cleaned = re.sub(r'[^0-9/]', '', entry_num)
    
    if '/' in cleaned:
        main, sub = cleaned.split('/')
        return (float(main), float(sub))
    return (float(cleaned), 0)

def sort_entries(entries: List[Entry]) -> List[Entry]:
    """Sort entries based on their numbers."""
    return sorted(entries, key=lambda x: parse_entry_number(x.number))