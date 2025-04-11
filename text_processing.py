from doctr.io import DocumentFile
from doctr.models import kie_predictor
import re
from typing import List, Tuple, Set
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

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
    """Check if text contains a valid poll number with enhanced pattern matching."""
    if not text:
        return False
    
    # Specific pattern matching for common register formats including dashes
    if re.match(r'^\d+(/\d+)?-*$', text) or re.match(r'^\d+(/\d+)?-*[A-Za-z]$', text) or re.match(r'^\d+[A-Za-z]$', text):
        return True
        
    cleaned = re.sub(r'[^0-9/]', '', text)
    if not cleaned:
        return False
        
    parts = cleaned.split('/')
    main_number = parts[0]
    
    # Reject if main number starts with 0
    if not main_number or main_number.startswith('0'):
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

def validate_entry_position(bbox: Tuple[float, float, float, float]) -> bool:
    """Validate that a bounding box position is reasonable."""
    x1, y1, x2, y2 = bbox
    
    # Check for reasonable box dimensions
    if x2 - x1 < 0.01 or y2 - y1 < 0.005:
        return False
        
    # Check if position is outside page bounds
    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        return False
        
    # Check for unreasonably large boxes (likely errors)
    if x2 - x1 > 0.5 or y2 - y1 > 0.1:
        return False
        
    return True

def validate_entry_name(name: str) -> bool:
    """Validate that an entry name is reasonable."""
    # Skip entries where the name is actually a poll number
    if re.match(r'^\d+(/\d+)?-*$', name.strip()):
        return False
        
    # Check if name has reasonable length
    words = name.split()
    if len(words) > 10 or len(words) == 0:
        return False
        
    # Check for common error patterns (large concatenated names)
    if len(name) > 50 or name.count(',') > 3:
        return False
        
    return True

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

def extract_entries_v1(poll_data: str) -> List[Entry]:
    """Extract entries with improved line-by-line processing."""
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
                    if name and validate_entry_name(name):
                        all_coords = [poll_coords] + name_coords
                        bbox = get_bounding_box(all_coords)
                        if bbox and validate_entry_position(bbox):
                            results.append(Entry(number=poll_num, name=name, bbox=bbox))
    
    return results

def extract_entries_v2(poll_data: str) -> List[Entry]:
    """Completely revised version of Method V2 with much stricter text grouping."""
    results = []
    data = eval(poll_data)
    
    # Sort by y-coordinate first to group by rows
    data_by_y = sorted(data, key=lambda x: x[1][0][1])
    
    # Get all x-coordinates to determine page dimensions
    x_coords = [entry[1][0][0] for entry in data_by_y]
    if not x_coords:
        return []
    
    # Calculate page width and split point for columns
    min_x = min(x_coords)
    max_x = max(x_coords)
    page_width = max_x - min_x
    middle_x = min_x + page_width / 2
    
    # Use much stricter line grouping
    lines = []
    current_line = []
    last_y = None
    
    # Use a very strict threshold for line separation
    y_threshold = 0.008  # Reduced from 0.01
    
    for text, coords in data_by_y:
        curr_y = coords[0][1]
        
        if last_y is None or abs(curr_y - last_y) > y_threshold:
            # Start a new line
            if current_line:
                lines.append(current_line)
            current_line = [(text, coords)]
        else:
            current_line.append((text, coords))
        
        last_y = curr_y
    
    if current_line:
        lines.append(current_line)
    
    # Process each line
    for line in lines:
        # Split line into left and right columns
        left_column = [entry for entry in line if entry[1][0][0] < middle_x]
        right_column = [entry for entry in line if entry[1][0][0] >= middle_x]
        
        # Sort each column by x-coordinate
        left_column.sort(key=lambda x: x[1][0][0])
        right_column.sort(key=lambda x: x[1][0][0])
        
        # Process each column separately
        for column in [left_column, right_column]:
            # Skip empty columns
            if not column:
                continue
            
            # Look for poll numbers in the column
            for i, (text, coords) in enumerate(column):
                # Only process if it's a valid poll number
                if is_number(text):
                    poll_num = text
                    name_parts = []
                    name_coords = []
                    
                    # Very strict limit on how many subsequent elements to check
                    # Only collect up to 4 elements for a name
                    name_limit = 4
                    
                    # Look at elements directly after the poll number
                    for j in range(i + 1, min(i + 1 + name_limit, len(column))):
                        next_text, next_coords = column[j]
                        
                        # Stop if we hit another poll number
                        if is_number(next_text):
                            break
                        
                        # Skip non-relevant elements
                        if next_text in ["-----", "----"] or re.match(r'^Page \d+$', next_text, re.IGNORECASE):
                            continue
                        
                        name_parts.append(next_text)
                        name_coords.append(next_coords)
                    
                    # Only create an entry if we found a name
                    if name_parts:
                        name = clean_text(' '.join(name_parts))
                        if name and validate_entry_name(name):
                            all_coords = [coords] + name_coords
                            bbox = get_bounding_box(all_coords)
                            if bbox and validate_entry_position(bbox):
                                results.append(Entry(number=poll_num, name=name, bbox=bbox))
    
    return results

def handle_letter_suffix(poll_num: str, column: List[Tuple[str, Tuple]], i: int) -> Tuple[str, int]:
    """Handle cases where a letter suffix appears as a separate element after the poll number."""
    if i + 1 < len(column) and re.match(r'^[A-Za-z]$', column[i+1][0]):
        letter = column[i+1][0]
        # Combine number and letter
        return f"{poll_num}{letter}", i + 1
    return poll_num, i

def validate_poll_number(number: str) -> bool:
    """Validate that a poll number format is legitimate."""
    # Check for poll numbers with a space before a letter (likely errors)
    if re.match(r'^\d+\s+[A-Za-z]$', number):
        return False
    return True

def extract_entries_grid(poll_data: str) -> List[Entry]:
    """Extract entries by first detecting the grid structure of the document."""
    data = eval(poll_data)
    results = []
    
    # Get all y-coordinates to find line positions
    y_coords = [(entry[1][0][1] + entry[1][1][1])/2 for entry in data]
    
    # Use clustering to identify distinct row positions
    rows = []
    current_row = []
    sorted_by_y = sorted(zip(y_coords, data), key=lambda x: x[0])
    
    y_threshold = 0.01  # Reduced from previous value for tighter line detection
    
    for i, (y, entry) in enumerate(sorted_by_y):
        if i == 0 or abs(y - sorted_by_y[i-1][0]) > y_threshold:
            if current_row:
                rows.append(current_row)
            current_row = [entry]
        else:
            current_row.append(entry)
    
    if current_row:
        rows.append(current_row)
    
    # Detect column structure based on x-coordinates
    for row_data in rows:
        # Sort by x-coordinate
        row_data.sort(key=lambda x: x[1][0][0])
        
        # Calculate the page width to determine column divisions
        x_coords = [entry[1][0][0] for entry in row_data]
        if not x_coords:
            continue
            
        min_x = min(x_coords)
        max_x = max(x_coords)
        page_width = max_x - min_x
        middle_x = min_x + page_width / 2
        
        # Process left and right columns separately
        left_column = [entry for entry in row_data if entry[1][0][0] < middle_x]
        right_column = [entry for entry in row_data if entry[1][0][0] >= middle_x]
        
        # Process each column
        for column in [left_column, right_column]:
            if not column:
                continue
                
            # Sort by x-coordinate within column
            column.sort(key=lambda x: x[1][0][0])
            
            # Look for poll numbers and associated names
            for i, (text, coords) in enumerate(column):
                if is_number(text):
                    poll_num = text
                    name_parts = []
                    name_coords = []
                    
                    # Collect name parts from subsequent grid cells in the same column
                    j = i + 1
                    while j < len(column) and not is_number(column[j][0]):
                        next_text = column[j][0]
                        
                        # Skip separator lines and page numbers
                        if not re.match(r'^[-]+$', next_text) and not re.match(r'^Page \d+$', next_text, re.IGNORECASE):
                            name_parts.append(next_text)
                            name_coords.append(column[j][1])
                        j += 1
                        
                        # Limit name collection to reasonable length
                        if len(name_parts) >= 5:
                            break
                    
                    if name_parts:
                        name = clean_text(' '.join(name_parts))
                        if name and validate_entry_name(name):
                            all_coords = [coords] + name_coords
                            bbox = get_bounding_box(all_coords)
                            if bbox and validate_entry_position(bbox):
                                results.append(Entry(number=poll_num, name=name, bbox=bbox))
    
    return results

def extract_entries_pattern(poll_data: str) -> List[Entry]:
    """Extract entries using common electoral register patterns with improved accuracy."""
    data = eval(poll_data)
    results = []
    
    # Sort by y-coordinate first to group entries by line
    data_by_y = sorted(data, key=lambda x: x[1][0][1])
    
    # Group into lines with adaptive threshold
    lines = []
    current_line = []
    last_y = None
    
    for text, coords in data_by_y:
        curr_y = coords[0][1]
        
        if last_y is None or abs(curr_y - last_y) > 0.01:  # Reduced from 0.012
            if current_line:
                lines.append(current_line)
            current_line = [(text, coords)]
        else:
            current_line.append((text, coords))
        
        last_y = curr_y
    
    if current_line:
        lines.append(current_line)
    
    # Process each line looking for different patterns
    for line in lines:
        # Sort by x-coordinate within each line
        line_sorted = sorted(line, key=lambda x: x[1][0][0])
        
        # Calculate page width for column detection
        x_coords = [entry[1][0][0] for entry in line_sorted]
        if not x_coords:
            continue
            
        min_x = min(x_coords)
        max_x = max(x_coords)
        page_width = max_x - min_x
        middle_x = min_x + page_width / 2
        
        # Split line into left and right columns
        left_column = [entry for entry in line_sorted if entry[1][0][0] < middle_x]
        right_column = [entry for entry in line_sorted if entry[1][0][0] >= middle_x]
        
        # Process each column separately
        for column in [left_column, right_column]:
            if not column:
                continue
                
            # Pattern 1: Number followed by name (most common)
            i = 0
            while i < len(column):
                text, coords = column[i]
                
                # Check for poll number patterns including variants with dashes
                if is_number(text):
                    poll_num = text
                    name_parts = []
                    name_coords = []
                    
                    # Look ahead for name parts until next poll number or end of column
                    j = i + 1
                    name_word_count = 0
                    while j < len(column):
                        next_text, next_coords = column[j]
                        
                        # Stop if we hit another poll number
                        if is_number(next_text):
                            break
                            
                        # Skip non-name elements like separator lines and page numbers
                        if not re.match(r'^[-]+$', next_text) and not re.match(r'^Page \d+$', next_text, re.IGNORECASE):
                            name_parts.append(next_text)
                            name_coords.append(next_coords)
                            name_word_count += 1
                            
                            # Limit name collection to reasonable length
                            if name_word_count >= 5:
                                break
                        j += 1
                    
                    # Create entry if valid
                    if name_parts:
                        name = clean_text(' '.join(name_parts))
                        if name and validate_entry_name(name):
                            all_coords = [coords] + name_coords
                            bbox = get_bounding_box(all_coords)
                            if bbox and validate_entry_position(bbox):
                                results.append(Entry(number=poll_num, name=name, bbox=bbox))
                                i = j  # Skip processed parts
                                continue
                
                # Check for special case: Number + Letter combination split into separate elements
                if re.match(r'^\d+(/\d+)?-*$', text) and i + 1 < len(column):
                    next_text, next_coords = column[i+1]
                    if re.match(r'^[A-Za-z]$', next_text):
                        # This might be a poll number with a letter suffix
                        poll_num = f"{text}{next_text}"
                        name_parts = []
                        name_coords = []
                        
                        # Look ahead for name parts starting from index i+2
                        j = i + 2
                        name_word_count = 0
                        while j < len(column):
                            next_text, next_coords = column[j]
                            
                            # Stop if we hit another poll number
                            if is_number(next_text):
                                break
                                
                            # Skip non-name elements like separator lines and page numbers
                            if not re.match(r'^[-]+$', next_text) and not re.match(r'^Page \d+$', next_text, re.IGNORECASE):
                                name_parts.append(next_text)
                                name_coords.append(next_coords)
                                name_word_count += 1
                                
                                # Limit name collection to reasonable length
                                if name_word_count >= 5:
                                    break
                            j += 1
                        
                        # Create entry if valid
                        if name_parts:
                            name = clean_text(' '.join(name_parts))
                            if name and validate_entry_name(name):
                                all_coords = [coords, next_coords] + name_coords
                                bbox = get_bounding_box(all_coords)
                                if bbox and validate_entry_position(bbox):
                                    results.append(Entry(number=poll_num, name=name, bbox=bbox))
                                    i = j  # Skip processed parts
                                    continue
                
                i += 1
    
    return results

def log_extraction_results(results, method_name, filepath="extraction_logs.txt"):
    """Log detailed information about extracted entries for a specific method."""
    with open(filepath, "a") as f:
        f.write(f"\n\n{'=' * 50}\n")
        f.write(f"METHOD: {method_name}\n")
        f.write(f"Total entries found: {len(results)}\n")
        f.write(f"{'=' * 50}\n\n")
        
        for i, entry in enumerate(results):
            f.write(f"Entry #{i+1}:\n")
            f.write(f"  Poll Number: {entry.number}\n")
            f.write(f"  Name: {entry.name}\n")
            f.write(f"  Position (bbox): {entry.bbox}\n")
            if entry.page is not None:
                f.write(f"  Page: {entry.page}\n")
            f.write(f"  {'_' * 40}\n\n")

def extract_entries_with_logging(poll_data: str) -> List[Entry]:
    """Enhanced version of extract_entries with detailed logging and improved algorithms."""
    # Clear previous log file
    with open("extraction_logs.txt", "w") as f:
        f.write("ELECTORAL REGISTER OCR EXTRACTION LOGS\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Run all extraction methods
    results_v1 = extract_entries_v1(poll_data)
    log_extraction_results(results_v1, "Method V1 (Line Processing)")
    
    results_v2 = extract_entries_v2(poll_data)
    log_extraction_results(results_v2, "Method V2 (Electoral Register Format)")
    
    results_grid = extract_entries_grid(poll_data)
    log_extraction_results(results_grid, "Method Grid (Grid Structure)")
    
    results_pattern = extract_entries_pattern(poll_data)
    log_extraction_results(results_pattern, "Method Pattern (Common Patterns)")
    
    # Combine all results
    all_results = results_v1 + results_v2 + results_grid + results_pattern
    
    # Create a dictionary to deduplicate entries
    result_dict = {}
    
    # Log information about deduplication process
    dedup_log = []
    
    # First pass: Store the cleanest entries by poll number
    for entry in all_results:
        # Standardize the poll number format
        poll_num = entry.number.strip()
        
        # Skip entries where the name is just a poll number
        if re.match(r'^\d+(/\d+)?-*$', entry.name.strip()):
            dedup_log.append(f"FILTERED: Poll #{poll_num}, Name: {entry.name} - Name is a poll number format")
            continue
        
        # Basic filtering for obvious errors
        # Only filter out entries where both poll number and name are problematic
        if (len(entry.name.split()) > 10 and  # Extremely long names
            not re.match(r'^[0-9]+(/[0-9]+)?-*[A-Za-z]?$', poll_num)):  # And non-standard poll numbers
            dedup_log.append(f"FILTERED: Poll #{poll_num}, Name: {entry.name} - Too long name and non-standard format")
            continue
        
        # Store in dictionary, preferring entries with cleaner formatting
        if poll_num in result_dict:
            existing = result_dict[poll_num]
            # Only replace if current entry has a significantly cleaner name
            name_words = len(entry.name.split())
            existing_words = len(existing.name.split())
            
            # If both entries have reasonable length names, keep the shorter one
            if name_words < existing_words and name_words <= 5:
                dedup_log.append(f"REPLACED: Poll #{poll_num} - Old: '{existing.name}' ({existing_words} words) -> New: '{entry.name}' ({name_words} words)")
                result_dict[poll_num] = entry
            else:
                dedup_log.append(f"KEPT: Poll #{poll_num} - Kept: '{existing.name}' ({existing_words} words) vs Candidate: '{entry.name}' ({name_words} words)")
        else:
            dedup_log.append(f"ADDED: Poll #{poll_num} - Name: '{entry.name}'")
            result_dict[poll_num] = entry
    
    # Second pass: Add entries with variant poll numbers that aren't duplicates
    variant_log = []
    for entry in all_results:
        poll_num = entry.number.strip()
        
        # Skip entries already processed or filtered
        if poll_num in result_dict:
            continue
            
        # Skip entries where the name is just a poll number
        if re.match(r'^\d+(/\d+)?-*$', entry.name.strip()):
            variant_log.append(f"FILTERED VARIANT: Poll #{poll_num}, Name: {entry.name} - Name is a poll number format")
            continue
            
        # Very minimal filtering - only skip entries with both problematic poll number and name
        if (len(entry.name.split()) > 10 and  # Extremely long names 
            not re.match(r'^[0-9]+(/[0-9]+)?-*[A-Za-z]?$', poll_num)):  # And non-standard poll numbers
            variant_log.append(f"FILTERED VARIANT: Poll #{poll_num}, Name: {entry.name} - Too long name and non-standard format")
            continue
        
        # For entries with poll numbers containing letters, extract the numeric part
        numeric_part = re.sub(r'[^0-9/]', '', poll_num)
        
        # Skip letter variants if they're likely errors
        if re.match(r'^\d+[A-Za-z]$', poll_num) and len(entry.name.split()) > 5:
            variant_log.append(f"FILTERED VARIANT: Poll #{poll_num}, Name: {entry.name} - Letter variant with suspicious long name")
            continue
        
        # Check if we already have an entry with the same numeric part
        has_numeric_match = False
        for existing_key in result_dict:
            existing_numeric = re.sub(r'[^0-9/]', '', existing_key)
            if numeric_part == existing_numeric:
                # Check if names are similar enough to be considered duplicates
                entry_name_lower = entry.name.lower()
                existing_name_lower = result_dict[existing_key].name.lower()
                
                # Only consider duplicate if there's substantial name overlap
                name_parts1 = set(entry_name_lower.split())
                name_parts2 = set(existing_name_lower.split())
                overlap = len(name_parts1.intersection(name_parts2))
                threshold = min(2, len(name_parts1) // 2, len(name_parts2) // 2)
                
                if overlap >= threshold:
                    has_numeric_match = True
                    variant_log.append(f"DUPLICATE VARIANT: Poll #{poll_num} (base: {existing_key}) - '{entry.name}' matches '{result_dict[existing_key].name}' with {overlap} common words")
                    break
        
        # Add entry if it's not a duplicate
        if not has_numeric_match and poll_num:
            variant_log.append(f"ADDED VARIANT: Poll #{poll_num} - Name: '{entry.name}'")
            result_dict[poll_num] = entry
    
    # Get all entries as a list
    results = list(result_dict.values())
    
    # Sort the results by poll number
    results.sort(key=lambda x: parse_entry_number(x.number))
    
    # Log deduplication process
    with open("extraction_logs.txt", "a") as f:
        f.write(f"\n\n{'=' * 50}\n")
        f.write(f"DEDUPLICATION PROCESS\n")
        f.write(f"{'=' * 50}\n\n")
        
        f.write("FIRST PASS (Main Deduplication):\n")
        for log_entry in dedup_log:
            f.write(f"  {log_entry}\n")
            
        f.write("\nSECOND PASS (Variant Processing):\n")
        for log_entry in variant_log:
            f.write(f"  {log_entry}\n")
    
    # Log final results
    log_extraction_results(results, "FINAL COMBINED RESULTS", "extraction_logs.txt")
    
    # Create a visual comparison of methods
    with open("extraction_comparison.txt", "w") as f:
        f.write("COMPARISON OF EXTRACTION METHODS\n\n")
        
        # Get all unique poll numbers across all methods
        all_numbers = set()
        for entries in [results_v1, results_v2, results_grid, results_pattern, results]:
            for entry in entries:
                all_numbers.add(entry.number.strip())
        
        # Sort poll numbers
        sorted_numbers = sorted(list(all_numbers), key=parse_entry_number)
        
        # Create dictionaries for easy lookup
        v1_dict = {entry.number.strip(): entry for entry in results_v1}
        v2_dict = {entry.number.strip(): entry for entry in results_v2}
        grid_dict = {entry.number.strip(): entry for entry in results_grid}
        pattern_dict = {entry.number.strip(): entry for entry in results_pattern}
        final_dict = {entry.number.strip(): entry for entry in results}
        
        # Create comparison table
        f.write(f"{'Poll#':<8} | {'Method V1':<5} | {'Method V2':<5} | {'Grid':<5} | {'Pattern':<5} | {'Final':<5} | Name\n")
        f.write(f"{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*30}\n")
        
        for num in sorted_numbers:
            v1_mark = "X" if num in v1_dict else " "
            v2_mark = "X" if num in v2_dict else " "
            grid_mark = "X" if num in grid_dict else " "
            pattern_mark = "X" if num in pattern_dict else " "
            final_mark = "X" if num in final_dict else " "
            
            name = ""
            if num in final_dict:
                name = final_dict[num].name
            elif num in pattern_dict:
                name = pattern_dict[num].name
            elif num in grid_dict:
                name = grid_dict[num].name
            elif num in v2_dict:
                name = v2_dict[num].name
            elif num in v1_dict:
                name = v1_dict[num].name
                
            f.write(f"{num:<8} | {v1_mark:^5} | {v2_mark:^5} | {grid_mark:^5} | {pattern_mark:^5} | {final_mark:^5} | {name[:30]}\n")
    
    return results

def parse_entry_number(entry_num: str) -> Tuple[float, float, float, float]:
    """Parse entry numbers into sortable tuples, with improved handling of variants."""
    try:
        # Handle letter suffixes like "231P"
        letter_suffix = ""
        if re.match(r'^\d+[A-Za-z]$', entry_num):
            letter_suffix = entry_num[-1]
            entry_num = entry_num[:-1]
        
        # Handle dash suffixes like "179-" or "151/1-"
        has_dash = False
        if entry_num.endswith('-'):
            has_dash = True
            entry_num = entry_num.rstrip('-')
        
        # Handle slash formats like "151/1"
        if '/' in entry_num:
            parts = entry_num.split('/')
            # Make sure parts[0] contains a valid number
            if not parts[0].strip().isdigit():
                return (0, 0, 0, 0)  # Return zeros for invalid numbers
                
            main = float(parts[0])
            
            # Handle empty second part or non-numeric second part
            if len(parts) > 1 and parts[1] and parts[1].strip().isdigit():
                sub = float(parts[1])
            else:
                sub = 0
                
            return (main, sub, ord(letter_suffix) if letter_suffix else 0, 1 if has_dash else 0)
        
        # Regular number - ensure it's actually a number before converting
        if not entry_num.strip().isdigit():
            return (0, 0, 0, 0)  # Return zeros for invalid numbers
            
        return (float(entry_num), 0, ord(letter_suffix) if letter_suffix else 0, 1 if has_dash else 0)
        
    except (ValueError, TypeError):
        # If any conversion fails, return zeros
        return (0, 0, 0, 0)
def get_main_number_safe(number_str: str) -> int:
    """Safely extract the main number from an entry."""
    try:
        # Remove letter suffixes and dashes
        clean_num = re.sub(r'[A-Za-z-], ', number_str)
        cleaned = re.sub(r'[^0-9/]', '', clean_num)
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

def extract_entries(poll_data: str) -> List[Entry]:
    """Main extraction function - use the improved version with logging."""
    return extract_entries_with_logging(poll_data)

def sort_entries(entries: List[Entry]) -> List[Entry]:
    """Sort entries based on their numbers."""
    return sorted(entries, key=lambda x: parse_entry_number(x.number))