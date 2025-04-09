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
    
    y_threshold = 0.01  # Adjust based on document spacing
    
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
            
        page_width = max(x_coords) - min(x_coords)
        middle_x = min(x_coords) + page_width / 2
        
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
                    
                    if name_parts:
                        name = clean_text(' '.join(name_parts))
                        if name:
                            all_coords = [coords] + name_coords
                            bbox = get_bounding_box(all_coords)
                            if bbox:
                                results.append(Entry(number=poll_num, name=name, bbox=bbox))
    
    return results


def extract_entries_pattern(poll_data: str) -> List[Entry]:
    """Extract entries using common electoral register patterns."""
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
        
        if last_y is None or abs(curr_y - last_y) > 0.012:  # Adjust threshold as needed
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
            
        page_width = max(x_coords) - min(x_coords)
        middle_x = min(x_coords) + page_width / 2
        
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
                
                # Check for poll number patterns including variants
                if re.match(r'^\d+(/\d+)?[A-Za-z]?$', text):
                    poll_num = text
                    name_parts = []
                    name_coords = []
                    
                    # Look ahead for name parts until next poll number or end of column
                    j = i + 1
                    while j < len(column):
                        next_text, next_coords = column[j]
                        
                        # Stop if we hit another poll number
                        if re.match(r'^\d+(/\d+)?[A-Za-z]?$', next_text):
                            break
                            
                        # Skip non-name elements like separator lines and page numbers
                        if not re.match(r'^[-]+$', next_text) and not re.match(r'^Page \d+$', next_text, re.IGNORECASE):
                            name_parts.append(next_text)
                            name_coords.append(next_coords)
                        j += 1
                    
                    # Create entry if valid
                    if name_parts:
                        name = clean_text(' '.join(name_parts))
                        if name:
                            all_coords = [coords] + name_coords
                            bbox = get_bounding_box(all_coords)
                            if bbox:
                                results.append(Entry(number=poll_num, name=name, bbox=bbox))
                                i = j  # Skip processed parts
                                continue
                
                # Check for special case: Number + Letter combination split into separate elements
                if re.match(r'^\d+$', text) and i + 1 < len(column):
                    next_text, next_coords = column[i+1]
                    if re.match(r'^[A-Za-z]$', next_text):
                        # This might be a poll number with a letter suffix
                        poll_num = f"{text}{next_text}"
                        name_parts = []
                        name_coords = []
                        
                        # Look ahead for name parts starting from index i+2
                        j = i + 2
                        while j < len(column):
                            next_text, next_coords = column[j]
                            
                            # Stop if we hit another poll number
                            if re.match(r'^\d+(/\d+)?[A-Za-z]?$', next_text):
                                break
                                
                            # Skip non-name elements like separator lines and page numbers
                            if not re.match(r'^[-]+$', next_text) and not re.match(r'^Page \d+$', next_text, re.IGNORECASE):
                                name_parts.append(next_text)
                                name_coords.append(next_coords)
                            j += 1
                        
                        # Create entry if valid
                        if name_parts:
                            name = clean_text(' '.join(name_parts))
                            if name:
                                all_coords = [coords, next_coords] + name_coords
                                bbox = get_bounding_box(all_coords)
                                if bbox:
                                    results.append(Entry(number=poll_num, name=name, bbox=bbox))
                                    i = j  # Skip processed parts
                                    continue
                
                # Pattern 2: Looking for poll numbers with specific spacing
                if re.match(r'^\d+$', text) or re.match(r'^\d+/\d+$', text):
                    poll_num = text
                    
                    # Check for consistent spacing pattern
                    if i + 1 < len(column):
                        next_x = column[i+1][1][0][0]
                        curr_x = coords[0][0]
                        x_spacing = next_x - curr_x
                        
                        # If spacing is within expected range for a name, process it
                        if 0.01 < x_spacing < 0.1:  # Adjust based on your document
                            name_candidates = []
                            
                            # Check subsequent elements
                            for j in range(i+1, len(column)):
                                next_text = column[j][0]
                                
                                # Stop if we hit another poll number
                                if re.match(r'^\d+(/\d+)?[A-Za-z]?$', next_text):
                                    break
                                    
                                # Skip non-name elements
                                if not re.match(r'^[-]+$', next_text) and not re.match(r'^Page \d+$', next_text, re.IGNORECASE):
                                    name_candidates.append((next_text, column[j][1]))
                            
                            if name_candidates:
                                name_parts = [text for text, _ in name_candidates]
                                name_coords = [coords for _, coords in name_candidates]
                                
                                name = clean_text(' '.join(name_parts))
                                if name:
                                    all_coords = [coords] + name_coords
                                    bbox = get_bounding_box(all_coords)
                                    if bbox:
                                        results.append(Entry(number=poll_num, name=name, bbox=bbox))
                                        i += len(name_candidates) + 1
                                        continue
                
                i += 1
    
    return results

def extract_entries(poll_data: str) -> List[Entry]:
    """Combined approach that integrates multiple extraction methods with minimal filtering."""
    # Run all extraction methods
    results_v1 = extract_entries_v1(poll_data)
    results_v2 = extract_entries_v2(poll_data)
    results_grid = extract_entries_grid(poll_data)
    results_pattern = extract_entries_pattern(poll_data)
    
    # Combine all results
    all_results = results_v1 + results_v2 + results_grid + results_pattern
    
    # Create a dictionary to deduplicate entries
    result_dict = {}
    
    # First pass: Store the cleanest entries by poll number
    for entry in all_results:
        # Standardize the poll number format
        poll_num = entry.number.strip()
        
        # Basic filtering for obvious errors
        # Only filter out entries where both poll number and name are problematic
        if (len(entry.name.split()) > 15 and  # Extremely long names
            not re.match(r'^[0-9]+(/[0-9]+)?[A-Za-z]?$', poll_num)):  # And non-standard poll numbers
            continue
        
        # Store in dictionary, preferring entries with cleaner formatting
        if poll_num in result_dict:
            existing = result_dict[poll_num]
            # Only replace if current entry has a significantly cleaner name
            name_words = len(entry.name.split())
            existing_words = len(existing.name.split())
            
            # If both entries have reasonable length names, keep the shorter one
            if name_words < existing_words and name_words <= 4:
                result_dict[poll_num] = entry
        else:
            result_dict[poll_num] = entry
    
    # Second pass: Add entries with variant poll numbers that aren't duplicates
    for entry in all_results:
        poll_num = entry.number.strip()
        
        # Skip entries already processed or filtered
        if poll_num in result_dict:
            continue
            
        # Very minimal filtering - only skip entries with both problematic poll number and name
        if (len(entry.name.split()) > 20 and  # Extremely long names 
            not re.match(r'^[0-9]+(/[0-9]+)?[A-Za-z]?$', poll_num)):  # And non-standard poll numbers
            continue
        
        # For entries with poll numbers containing letters, extract the numeric part
        numeric_part = re.sub(r'[^0-9/]', '', poll_num)
        
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
                if len(name_parts1.intersection(name_parts2)) >= min(2, len(name_parts1) // 2, len(name_parts2) // 2):
                    has_numeric_match = True
                    break
        
        # Add entry if it's not a duplicate
        if not has_numeric_match and poll_num:
            result_dict[poll_num] = entry
    
    # Get all entries as a list
    results = list(result_dict.values())
    
    # Sort the results by poll number
    results.sort(key=lambda x: parse_entry_number(x.number))
    
    # Log extraction statistics
    with open("extraction_results.txt", "w") as f:
        f.write(f"Version 1 found {len(results_v1)} entries\n")
        f.write(f"Version 2 found {len(results_v2)} entries\n")
        f.write(f"Grid-based found {len(results_grid)} entries\n")
        f.write(f"Pattern-based found {len(results_pattern)} entries\n")
        f.write(f"Combined approach has {len(results)} entries\n\n")
        
        # Log all poll numbers
        for entry in results:
            f.write(f"{entry.number}: {entry.name}\n")
    
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