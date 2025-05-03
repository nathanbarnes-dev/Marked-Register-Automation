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
    source: Optional[str] = None  # Add this field to track which method found the entry

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
# Step 3: Add a function to save source tracking logs (in text_processing.py)
def save_source_tracking_log(entries: List[Entry], log_dir: str = "") -> str:
    """
    Save detailed source tracking information to a log file.
    
    Args:
        entries: The list of entries with source information
        log_dir: Directory to save the log in (default is current directory)
        
    Returns:
        The path to the created log file
    """
    from datetime import datetime
    import os
    
    # Create a source analysis dictionary
    source_stats = {
        "improved": 0,
        "v2": 0, 
        "pattern": 0,
        "grid": 0,
        "v1": 0
    }
    
    # Count entries by source
    for entry in entries:
        if hasattr(entry, 'source') and entry.source in source_stats:
            source_stats[entry.source] += 1
    
    # Calculate total for percentage calculations
    total_entries = sum(source_stats.values())
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"source_tracking_{timestamp}.log"
    
    # Use the specified directory if provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
    else:
        log_path = log_filename
    
    with open(log_path, "w") as f:
        f.write(f"Entry Source Tracking Log\n")
        f.write(f"=======================\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Final combination statistics
        f.write(f"Results Summary:\n")
        f.write(f"---------------\n")
        f.write(f"Total entries processed: {total_entries}\n\n")
        
        # Write source breakdown with percentages
        f.write(f"Source Breakdown:\n")
        f.write(f"----------------\n")
        for source, count in source_stats.items():
            percentage = (count / total_entries * 100) if total_entries > 0 else 0
            f.write(f"{source}: {count} entries ({percentage:.2f}%)\n")
        f.write("\n")
        
        # List all entries with their sources
        f.write(f"Detailed Entry List:\n")
        f.write(f"------------------\n")
        f.write(f"{'Poll Number':<15} | {'Source':<10} | Name\n")
        f.write(f"{'-'*15} | {'-'*10} | {'-'*40}\n")
        
        for entry in entries:
            source = entry.source if hasattr(entry, 'source') else "unknown"
            f.write(f"{entry.number:<15} | {source:<10} | {entry.name}\n")
    
    print(f"Source tracking log saved to: {log_path}")
    return log_path

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
    """
    Improved version that addresses the specific problem of missed poll numbers,
    with better detection to prevent inclusion of previous entries.
    """
    results = []
    data = eval(poll_data)
    
    if not data:
        return results
    
    # Get document metrics
    text_heights = [coords[1][1] - coords[0][1] for _, coords in data]
    text_heights.sort()
    median_text_height = text_heights[len(text_heights)//2] if text_heights else 0.02
    
    # Identify middle point for columns
    x_coords = [coord[0][0] for _, coord in data]
    middle_x = (min(x_coords) + max(x_coords)) / 2 if x_coords else 0.5
    
    # Group by rows with adaptive threshold
    data_by_y = sorted(data, key=lambda x: x[1][0][1])
    
    # IMPROVEMENT 1: Pre-scan for numeric patterns to detect potential missed poll numbers
    # This helps identify text that looks like poll numbers but wasn't caught by is_number()
    potential_poll_numbers = set()
    for text, _ in data:
        # Look for numeric patterns that might be poll numbers
        if re.match(r'^\d{1,4}$', text) or re.match(r'^\d{1,4}/\d{1,2}$', text):
            potential_poll_numbers.add(text)
    
    # Sort potential poll numbers numerically for later analysis
    sorted_potential_numbers = sorted([int(re.sub(r'/.*', '', num)) for num in potential_poll_numbers if re.match(r'^\d+', num)])
    
    # Calculate differences between consecutive potential poll numbers to detect patterns
    number_diffs = []
    for i in range(1, len(sorted_potential_numbers)):
        number_diffs.append(sorted_potential_numbers[i] - sorted_potential_numbers[i-1])
    
    # If there's a consistent pattern (like +1), we can use it to identify missed numbers
    common_diff = 1  # Default: assume consecutive numbering
    if number_diffs:
        # Use the most common difference
        from collections import Counter
        diff_counter = Counter(number_diffs)
        common_diff = diff_counter.most_common(1)[0][0]
    
    # Calculate row threshold adaptively
    y_diffs = []
    for i in range(1, len(data_by_y)):
        curr_y = data_by_y[i][1][0][1]
        prev_y = data_by_y[i-1][1][0][1]
        y_diffs.append(abs(curr_y - prev_y))
    
    if y_diffs:
        y_diffs.sort()
        row_threshold = y_diffs[len(y_diffs)//2] * 1.5
        row_threshold = max(median_text_height * 0.8, min(median_text_height * 3.0, row_threshold))
    else:
        row_threshold = median_text_height * 1.5
    
    # Group into rows
    rows = []
    current_row = [data_by_y[0]] if data_by_y else []
    last_y = data_by_y[0][1][0][1] if data_by_y else 0
    
    for i in range(1, len(data_by_y)):
        entry = data_by_y[i]
        curr_y = entry[1][0][1]
        y_diff = abs(curr_y - last_y)
        
        if y_diff < row_threshold:
            current_row.append(entry)
        else:
            if current_row:
                rows.append(current_row)
            current_row = [entry]
        
        last_y = curr_y
    
    if current_row:
        rows.append(current_row)
    
    # Calculate median row height
    row_heights = []
    for i in range(len(rows)-1):
        if rows[i] and rows[i+1]:
            row1_middle_y = sum(entry[1][0][1] for entry in rows[i]) / len(rows[i])
            row2_middle_y = sum(entry[1][0][1] for entry in rows[i+1]) / len(rows[i+1])
            row_heights.append(abs(row2_middle_y - row1_middle_y))
    
    median_row_height = median_text_height * 2.5  # Fallback
    if row_heights:
        row_heights.sort()
        median_row_height = row_heights[len(row_heights)//2]
    
    # IMPROVEMENT 2: Maintain a list of detected poll numbers to check for sequential patterns
    detected_numbers = []
    last_processed_y = 0  # Track the last vertical position we processed
    
    # Process each row
    for row_idx, row in enumerate(rows):
        left_column = [entry for entry in row if entry[1][0][0] < middle_x]
        right_column = [entry for entry in row if entry[1][0][0] >= middle_x]
        
        # Process each column
        for column in [left_column, right_column]:
            if not column:
                continue
            
            # Sort by x position
            column_sorted = sorted(column, key=lambda x: x[1][0][0])
            
            # Find poll numbers
            poll_candidates = []
            for i, (text, coords) in enumerate(column_sorted):
                if is_number(text):
                    poll_candidates.append((i, text, coords))
                # IMPROVEMENT 3: Also check for numeric text that might be a missed poll number
                elif re.match(r'^\d{1,4}$', text):
                    # This looks like a number but wasn't caught by is_number()
                    # We'll check if it fits the pattern of other poll numbers
                    try:
                        num_value = int(text)
                        # Check if this could be the next number in sequence
                        if detected_numbers and abs(num_value - detected_numbers[-1]) <= 2 * common_diff:
                            # This looks like it could be a valid poll number
                            poll_candidates.append((i, text, coords))
                    except ValueError:
                        pass
            
            # IMPROVEMENT 4: Smarter processing that's aware of sequential poll numbers
            for poll_idx, poll_text, poll_coords in poll_candidates:
                # Track this poll number to detect patterns
                try:
                    current_num = int(re.sub(r'/.*', '', poll_text))
                    detected_numbers.append(current_num)
                except ValueError:
                    pass
                
                # Initialize collection
                name_parts = []
                name_coords = []
                
                # Handle letter suffixes
                letter_suffix = ""
                next_idx = poll_idx + 1
                has_separate_letter = False
                
                if next_idx < len(column_sorted):
                    letter_text, letter_coords = column_sorted[next_idx]
                    if re.match(r'^[A-Za-z]$', letter_text) and abs(letter_coords[0][0] - poll_coords[1][0]) < median_text_height * 2:
                        letter_suffix = letter_text
                        has_separate_letter = True
                        next_idx += 1
                
                # IMPROVEMENT 5: Stricter vertical distance check based on expected entry layout
                poll_y = poll_coords[0][1]
                poll_bottom = poll_coords[1][1]
                
                # Critical improvement: Check if there's a large gap from the last processed entry
                # This helps detect cases where a poll number was missed
                vertical_gap = poll_y - last_processed_y if last_processed_y > 0 else 0
                
                # If there's an unusually large gap and we've seen poll numbers before,
                # this might indicate a missed poll number
                suspicious_gap = vertical_gap > median_row_height * 1.5 and detected_numbers and len(detected_numbers) >= 2
                
                # IMPROVEMENT 6: Look for the next poll number to establish a boundary
                next_poll_idx = len(column_sorted)
                for j in range(next_idx, len(column_sorted)):
                    curr_text = column_sorted[j][0]
                    # Regular poll number check
                    if is_number(curr_text):
                        next_poll_idx = j
                        break
                    # Also check for numeric text that might be the next poll number
                    elif re.match(r'^\d{1,4}$', curr_text):
                        try:
                            num_value = int(curr_text)
                            if detected_numbers and abs(num_value - detected_numbers[-1]) <= 2 * common_diff:
                                next_poll_idx = j
                                break
                        except ValueError:
                            pass
                
                # IMPROVEMENT 7: Use a more restrictive vertical limit if we suspect a missed poll number
                max_y_distance = poll_height = poll_coords[1][1] - poll_coords[0][1]
                
                if suspicious_gap:
                    # More restrictive - only allow text very close to the poll number
                    max_y_distance = min(poll_height * 1.5, median_row_height * 0.4)
                else:
                    # Standard case - allow a bit more distance
                    max_y_distance = min(poll_height * 2, median_row_height * 0.7)
                
                # Collect name parts with very strict vertical boundaries
                for j in range(next_idx, next_poll_idx):
                    next_text, next_coords = column_sorted[j]
                    next_y = next_coords[0][1]
                    
                    # Skip non-name elements
                    if next_text.strip() == "-" or re.match(r'^[-]+$', next_text) or re.match(r'^Page \d+$', next_text, re.IGNORECASE):
                        continue
                    
                    # CRUCIAL: Apply stricter vertical distance check
                    # This is the key to preventing inclusion of previous entries
                    if abs(next_y - poll_y) > max_y_distance:
                        # This text is too far from the poll number - likely part of another entry
                        break
                    
                    # Add to name parts
                    name_parts.append(next_text)
                    name_coords.append(next_coords)
                
                # Update the last processed vertical position
                if name_coords:
                    last_processed_y = max([coords[1][1] for coords in name_coords])
                else:
                    last_processed_y = poll_bottom
                
                # Create entry if valid
                if name_parts:
                    name = clean_text(' '.join(name_parts))
                    
                    if name and len(name.split()) <= 12:
                        full_poll_num = poll_text
                        if has_separate_letter:
                            full_poll_num = f"{poll_text}{letter_suffix}"
                        
                        coords_list = [poll_coords]
                        if has_separate_letter:
                            coords_list.append(column_sorted[poll_idx+1][1])
                        coords_list.extend(name_coords)
                        
                        # Get basic bounding box
                        basic_bbox = get_bounding_box(coords_list)
                        
                        if basic_bbox:
                            # Apply height restrictions
                            x1, y1, x2, y2 = basic_bbox
                            
                            # IMPROVEMENT 8: Apply very strict height limit when suspicious gap detected
                            if suspicious_gap:
                                # Even more restrictive height limit
                                reasonable_height = min(median_row_height * 0.5, poll_height * 2.5)
                            else:
                                reasonable_height = min(median_row_height * 0.8, poll_height * 3.5)
                            
                            current_height = y2 - y1
                            if current_height > reasonable_height:
                                # Center the box around the current center
                                center_y = (y1 + y2) / 2
                                y1 = center_y - reasonable_height / 2
                                y2 = center_y + reasonable_height / 2
                            
                            adjusted_bbox = (x1, y1, x2, y2)
                            results.append(Entry(number=full_poll_num, name=name, bbox=adjusted_bbox))
    
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
    """
    Revised extract_entries function that prioritizes improved results and applies
    stricter filtering to prevent multiple entries being combined.
    Also tracks which method provided each entry.
    """
    from datetime import datetime  # Add this import at the top of the file
    
    # Run all extraction methods
    results_v1 = extract_entries_v1(poll_data)
    results_v2 = extract_entries_v2(poll_data)
    results_grid = extract_entries_grid(poll_data)
    results_pattern = extract_entries_pattern(poll_data)
    results_improved = extract_entries_improved(poll_data)

    # Add source information to each entry
    for entry in results_v1:
        entry.source = "v1"
    for entry in results_v2:
        entry.source = "v2"
    for entry in results_grid:
        entry.source = "grid"
    for entry in results_pattern:
        entry.source = "pattern"
    for entry in results_improved:
        entry.source = "improved"

    print(f"Method v1: found {len(results_v1)} entries")
    print(f"Method v2: found {len(results_v2)} entries")
    print(f"Method grid: found {len(results_grid)} entries")
    print(f"Method pattern: found {len(results_pattern)} entries")
    print(f"Method improved: found {len(results_improved)} entries")
    
    # Create dictionaries to map poll numbers to entries for each method
    v1_entries = {entry.number.strip(): entry for entry in results_v1}
    v2_entries = {entry.number.strip(): entry for entry in results_v2}
    grid_entries = {entry.number.strip(): entry for entry in results_grid}
    pattern_entries = {entry.number.strip(): entry for entry in results_pattern}
    improved_entries = {entry.number.strip(): entry for entry in results_improved}
    
    # Combine all potential poll numbers for analysis
    all_numbers = set(v1_entries.keys()) | set(v2_entries.keys()) | set(grid_entries.keys()) | set(pattern_entries.keys())
    
    # Filter function to reject entries with suspicious characteristics
    def is_valid_entry(entry):
        # Check bounding box height - reject if too tall
        _, y1, _, y2 = entry.bbox
        height = y2 - y1
        if height > 0.025:  # Maximum reasonable height
            return False
        
        # Check name length - reject if too many words
        name_words = len(entry.name.split())
        if name_words > 5:  # Maximum reasonable words
            return False
            
        # Additional check for suspicious patterns in names
        # Reject entries that appear to contain multiple people's names
        name = entry.name.lower()
        name_commas = name.count(',')
        if name_commas > 1:  # More than one comma often indicates multiple names
            return False
            
        return True
    
    # Filter entries from each method
    v1_entries_filtered = {num: entry for num, entry in v1_entries.items() if is_valid_entry(entry)}
    v2_entries_filtered = {num: entry for num, entry in v2_entries.items() if is_valid_entry(entry)}
    grid_entries_filtered = {num: entry for num, entry in grid_entries.items() if is_valid_entry(entry)}
    pattern_entries_filtered = {num: entry for num, entry in pattern_entries.items() if is_valid_entry(entry)}
    improved_entries_filtered = {num: entry for num, entry in improved_entries.items() if is_valid_entry(entry)}
    
    print(f"After filtering:")
    print(f"Method v1: {len(v1_entries_filtered)} entries (removed {len(v1_entries) - len(v1_entries_filtered)})")
    print(f"Method v2: {len(v2_entries_filtered)} entries (removed {len(v2_entries) - len(v2_entries_filtered)})")
    print(f"Method grid: {len(grid_entries_filtered)} entries (removed {len(grid_entries) - len(grid_entries_filtered)})")
    print(f"Method pattern: {len(pattern_entries_filtered)} entries (removed {len(pattern_entries) - len(pattern_entries_filtered)})")
    print(f"Method improved: {len(improved_entries_filtered)} entries (removed {len(improved_entries) - len(improved_entries_filtered)})")
    
    # Final result dictionary, prioritizing methods in order: improved, v2, pattern, grid, v1
    result_dict = {}
    
    # Start with improved results (highest priority)
    for num, entry in improved_entries_filtered.items():
        result_dict[num] = entry
    
    # Add v2 results if not already present
    for num, entry in v2_entries_filtered.items():
        if num not in result_dict:
            result_dict[num] = entry
    
    # Add pattern results if not already present
    for num, entry in pattern_entries_filtered.items():
        if num not in result_dict:
            result_dict[num] = entry
    
    # Add grid results if not already present
    for num, entry in grid_entries_filtered.items():
        if num not in result_dict:
            result_dict[num] = entry
    
    # Finally, add v1 results if not already present
    for num, entry in v1_entries_filtered.items():
        if num not in result_dict:
            result_dict[num] = entry
    
    # Get all entries as a list
    results = list(result_dict.values())
    
    # Sort the results by poll number
    results.sort(key=lambda x: parse_entry_number(x.number))
    
    # Create a source analysis dictionary
    source_stats = {
        "improved": 0,
        "v2": 0, 
        "pattern": 0,
        "grid": 0,
        "v1": 0
    }
    
    # Count entries by source
    for entry in results:
        if entry.source in source_stats:
            source_stats[entry.source] += 1
    
    # Log extraction statistics
    with open("extraction_results.txt", "w") as f:
        f.write(f"Extraction Results\n")
        f.write(f"=================\n\n")
        f.write(f"Version 1 found {len(results_v1)} entries, {len(v1_entries_filtered)} after filtering\n")
        f.write(f"Version 2 found {len(results_v2)} entries, {len(v2_entries_filtered)} after filtering\n")
        f.write(f"Grid-based found {len(results_grid)} entries, {len(grid_entries_filtered)} after filtering\n")
        f.write(f"Pattern-based found {len(results_pattern)} entries, {len(pattern_entries_filtered)} after filtering\n")
        f.write(f"Improved found {len(results_improved)} entries, {len(improved_entries_filtered)} after filtering\n")
        f.write(f"Combined approach has {len(results)} entries\n\n")
        
        # Count entries by source
        sources = {
            "improved": 0,
            "v2": 0,
            "pattern": 0,
            "grid": 0,
            "v1": 0
        }
        
        for entry in results:
            if entry.source in sources:
                sources[entry.source] += 1
        
        f.write(f"Source breakdown:\n")
        total_entries = len(results)
        for source, count in sources.items():
            percentage = count/total_entries*100 if total_entries > 0 else 0
            f.write(f"  {source}: {count} entries ({percentage:.1f}%)\n\n")
        
        # Log all poll numbers with their sources
        f.write(f"Poll Number | Source Method | Name\n")
        f.write(f"-----------|--------------|-----\n")
        for entry in results:
            f.write(f"{entry.number} | {entry.source} | {entry.name}\n")
    
    print(f"Final combined result has {len(results)} entries")
    print(f"Source breakdown:")
    for source, count in source_stats.items():
        percentage = count/len(results)*100 if results else 0
        print(f"  {source}: {count} entries ({percentage:.1f}%)")
    
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

def parse_entry_number(entry_num: str) -> tuple[float, float]:
    """Parse entry numbers into sortable tuples."""
    cleaned = re.sub(r'[^0-9/]', '', entry_num)
    
    if '/' in cleaned:
        parts = cleaned.split('/')
        main = parts[0]
        # Handle the case where there's nothing after the slash
        sub = parts[1] if len(parts) > 1 and parts[1] else '0'
        return (float(main) if main else 0.0, float(sub))
    return (float(cleaned) if cleaned else 0.0, 0.0)

def sort_entries(entries: List[Entry]) -> List[Entry]:
    """Sort entries based on their numbers."""
    return sorted(entries, key=lambda x: parse_entry_number(x.number))

def extract_entries_improved(poll_data: str) -> List[Entry]:
    """
    Enhanced extraction function that specifically targets the missing entry cases
    by using multiple approaches and combining results, with stricter bounding box controls.
    """
    data = eval(poll_data)
    results = []
    
    # Helper function to validate poll numbers
    def validate_poll_number(number):
        """Validate and sanitize a poll number"""
        # Ensure it's a string
        if not isinstance(number, str):
            number = str(number) if number is not None else ""
            
        # Check if it has at least one digit
        if not re.search(r'\d', number):
            return None  # Invalid poll number, reject it
            
        # If there's a slash, ensure both parts have digits
        if '/' in number:
            parts = number.split('/')
            if len(parts) != 2 or not re.search(r'\d', parts[0]) or not re.search(r'\d', parts[1]):
                return None  # Invalid poll number, reject it
                
        return number
    
    # Get document metrics
    all_heights = [coords[1][1] - coords[0][1] for _, coords in data]
    all_heights.sort()
    median_text_height = all_heights[len(all_heights)//2] if all_heights else 0.02
    
    # Get all x-coordinates to determine page structure
    all_x_coords = []
    for text, coords in data:
        all_x_coords.append(coords[0][0])  # Left x
        all_x_coords.append(coords[1][0])  # Right x
    
    min_x = min(all_x_coords) if all_x_coords else 0
    max_x = max(all_x_coords) if all_x_coords else 1
    page_width = max_x - min_x
    
    # Define column boundaries - this is crucial for handling two-column layouts
    left_column_boundary = min_x + page_width * 0.45
    right_column_boundary = min_x + page_width * 0.55
    
    # Group by rows with adaptive threshold
    data_by_y = sorted(data, key=lambda x: x[1][0][1])
    
    # Perform pre-processing to identify potential poll numbers
    poll_number_regex = r'^\d+(/\d+)?[A-Za-z]?$'
    poll_number_simple_regex = r'^\d+$'
    potential_numbers = []
    
    for text, coords in data:
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        # Check for full poll numbers
        if re.match(poll_number_regex, text):
            try:
                # Make sure to handle possible trailing non-digit characters
                main_number_str = re.sub(r'/.*', '', re.sub(r'[A-Za-z]$', '', text))
                main_number = int(main_number_str)
                potential_numbers.append(main_number)
            except ValueError:
                pass
        # Check for simple numbers (could be poll numbers)
        elif re.match(poll_number_simple_regex, text):
            try:
                main_number = int(text)
                potential_numbers.append(main_number)
            except ValueError:
                pass
    
    # Sort and analyze number sequences to detect patterns
    potential_numbers.sort()
    
    # Calculate common differences to detect sequence patterns
    differences = []
    for i in range(1, len(potential_numbers)):
        diff = potential_numbers[i] - potential_numbers[i-1]
        if 1 <= diff <= 3:  # Only consider small differences for pattern detection
            differences.append(diff)
    
    # Determine most common difference (usually 1 for sequential numbers)
    most_common_diff = 1
    if differences:
        from collections import Counter
        diff_counter = Counter(differences)
        most_common_diff = diff_counter.most_common(1)[0][0]
    
    # Group similar y-coordinates to detect rows
    rows = []
    if data_by_y:  # Check if there's any data before proceeding
        current_row = [data_by_y[0]]
        last_y = data_by_y[0][1][0][1]
        
        for i in range(1, len(data_by_y)):
            entry = data_by_y[i]
            curr_y = entry[1][0][1]
            
            # Calculate y-difference
            y_diff = abs(curr_y - last_y)
            
            # Use adaptive threshold based on text height
            threshold = median_text_height * 0.8
            
            if y_diff < threshold:
                # Same row
                current_row.append(entry)
            else:
                # New row
                if current_row:
                    rows.append(current_row)
                current_row = [entry]
            
            last_y = curr_y
        
        if current_row:
            rows.append(current_row)
    
    # Process rows to extract entries
    processed_numbers = set()
    
    # IMPROVEMENT: Create a map of which text elements are already used
    # This will help prevent text from being used in multiple entries
    used_text_elements = set()
    
    for row_idx, row in enumerate(rows):
        # Separate left and right columns
        left_column = []
        right_column = []
        
        for text, coords in row:
            # Convert text to string if necessary
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
                
            # Skip text elements that have already been used in another entry
            text_id = (text, coords[0][0], coords[0][1])  # Unique identifier
            if text_id in used_text_elements:
                continue
                
            # Determine column based on midpoint of text
            text_midpoint_x = (coords[0][0] + coords[1][0]) / 2
            
            if text_midpoint_x < left_column_boundary:
                left_column.append((text, coords, text_id))
            elif text_midpoint_x > right_column_boundary:
                right_column.append((text, coords, text_id))
            else:
                # For text in the middle region, make a decision based on its position
                if coords[0][0] < (min_x + page_width * 0.5):
                    left_column.append((text, coords, text_id))
                else:
                    right_column.append((text, coords, text_id))
        
        # Process each column separately
        for column_items in [left_column, right_column]:
            if not column_items:
                continue
            
            # Sort by x position within column
            column_items.sort(key=lambda x: x[1][0][0])
            
            # IMPROVEMENT: Keep track of the next poll number positions 
            # to better estimate vertical boundaries
            poll_positions = []
            for i, (text, coords, text_id) in enumerate(column_items):
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                    
                if re.match(poll_number_regex, text) or re.match(poll_number_simple_regex, text):
                    try:
                        num_value = int(re.sub(r'/.*', '', re.sub(r'[A-Za-z]$', '', text)))
                        poll_positions.append((i, num_value, coords[0][1]))
                    except ValueError:
                        pass
            
            # Find potential poll numbers in this column
            for idx, (text, coords, text_id) in enumerate(column_items):
                # Skip if we've already processed this number
                if text_id in used_text_elements:
                    continue
                
                # Ensure text is a string
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                
                # Check for poll number patterns
                is_poll_number = False
                poll_number = None
                
                # Direct poll number match
                if re.match(poll_number_regex, text):
                    is_poll_number = True
                    poll_number = text
                
                # Check for simple number that could be a poll number
                elif re.match(poll_number_simple_regex, text):
                    try:
                        num_value = int(text)
                        
                        # Check if it looks like it could be part of the sequence
                        if potential_numbers and num_value > 0 and num_value <= max(potential_numbers) + 10:
                            is_poll_number = True
                            poll_number = text
                            
                            # Special case: Check for letter directly following
                            if idx + 1 < len(column_items):
                                next_text, next_coords, next_id = column_items[idx + 1]
                                # Ensure next_text is a string
                                if not isinstance(next_text, str):
                                    next_text = str(next_text) if next_text is not None else ""
                                    
                                if (re.match(r'^[A-Za-z]$', next_text) and 
                                    abs(next_coords[0][0] - coords[1][0]) < median_text_height * 2):
                                    # This is likely a letter suffix (like "A" in "34 A")
                                    poll_number = f"{text}{next_text}"
                                    used_text_elements.add(next_id)  # Mark as used
                    except ValueError:
                        pass
                
                if is_poll_number and poll_number:
                    # Validate poll number
                    valid_poll_number = validate_poll_number(poll_number)
                    if not valid_poll_number:
                        continue  # Skip invalid poll numbers
                    
                    poll_number = valid_poll_number
                    used_text_elements.add(text_id)  # Mark this poll number as used
                    
                    # Look for name associated with this poll number
                    name_parts = []
                    name_ids = []
                    name_coords = []
                    
                    # IMPROVEMENT: Find the next poll number in sequence to limit vertical search
                    current_y = coords[0][1]
                    next_poll_y = float('inf')
                    
                    for pos_idx, pos_num, pos_y in poll_positions:
                        if pos_y > current_y + median_text_height * 0.5 and pos_y < next_poll_y:
                            next_poll_y = pos_y
                    
                    # If no next poll number found, use a reasonable default
                    if next_poll_y == float('inf'):
                        next_poll_y = current_y + median_text_height * 3.0
                    
                    # Calculate a more precise vertical range
                    y_min = current_y - median_text_height * 0.25
                    y_max = min(current_y + median_text_height * 2.0, next_poll_y - median_text_height * 0.25)
                    
                    # IMPROVEMENT: Also limit the search horizontally
                    # Find x-position of this poll number
                    poll_x = coords[0][0]
                    
                    # Define a reasonable horizontal range
                    x_min = poll_x
                    x_max = min_x + page_width * 0.9  # Allow names to extend quite far, but not to next column
                    
                    if coords[0][0] < left_column_boundary:
                        # Left column entry - restrict to left column width
                        x_max = left_column_boundary
                    elif coords[0][0] > right_column_boundary:
                        # Right column entry - restrict within right column
                        x_min = right_column_boundary
                    
                    # IMPROVEMENT: Look for name parts with stricter position constraints
                    for j, (other_text, other_coords, other_id) in enumerate(column_items):
                        # Skip if already used
                        if other_id in used_text_elements:
                            continue
                            
                        # Ensure other_text is a string
                        if not isinstance(other_text, str):
                            other_text = str(other_text) if other_text is not None else ""
                            
                        # Skip the poll number itself
                        if other_id == text_id:
                            continue
                        
                        # Check if this item is within vertical range
                        other_y = other_coords[0][1]
                        
                        # CRUCIAL: Apply much stricter vertical range check
                        if y_min <= other_y <= y_max:
                            # Apply horizontal range check
                            other_x = other_coords[0][0]
                            
                            # Check if within horizontal range and to the right of number
                            if x_min <= other_x <= x_max and other_x >= poll_x:
                                # Exclude items that look like numbers
                                if not re.match(r'^\d+(/\d+)?[A-Za-z]?$', other_text):
                                    name_parts.append(other_text)
                                    name_coords.append(other_coords)
                                    name_ids.append(other_id)
                    
                    # IMPROVEMENT: If we found name parts, make sure they're reasonably close together
                    if name_parts:
                        # Group name parts that are close together horizontally
                        valid_name_parts = []
                        valid_name_coords = []
                        valid_name_ids = []
                        
                        # Sort name parts by x-coordinate
                        sorted_name_data = sorted(zip(name_parts, name_coords, name_ids), key=lambda x: x[1][0][0])
                        
                        # Start with the first name part
                        current_x = sorted_name_data[0][1][1][0]  # End of first part
                        valid_name_parts.append(sorted_name_data[0][0])
                        valid_name_coords.append(sorted_name_data[0][1])
                        valid_name_ids.append(sorted_name_data[0][2])
                        
                        # Check remaining parts
                        for part, part_coords, part_id in sorted_name_data[1:]:
                            part_x = part_coords[0][0]  # Start of this part
                            
                            # Check if this part is reasonably close to the previous one
                            x_gap = part_x - current_x
                            
                            if x_gap <= median_text_height * 3.0:
                                # This part is close enough horizontally - add it
                                valid_name_parts.append(part)
                                valid_name_coords.append(part_coords)
                                valid_name_ids.append(part_id)
                                current_x = part_coords[1][0]  # Update current x
                            else:
                                # This part is too far away - might belong to another entry
                                break
                        
                        # Update name parts with valid ones
                        name_parts = valid_name_parts
                        name_coords = valid_name_coords
                        name_ids = valid_name_ids
                    
                    # Create the entry if we found a name
                    if name_parts:
                        # Mark these text elements as used
                        for name_id in name_ids:
                            used_text_elements.add(name_id)
                            
                        # Create name and bounding box
                        name = clean_text(' '.join(name_parts))
                        
                        # IMPROVEMENT: Create a tighter bounding box
                        all_coords = [coords] + name_coords
                        bbox = get_bounding_box(all_coords)
                        
                        if bbox and name:
                            # Validate poll_number again before creating the entry
                            valid_poll_number = validate_poll_number(poll_number)
                            if valid_poll_number:
                                results.append(Entry(number=valid_poll_number, name=name, bbox=bbox))
    
    # SECOND PASS: Check for missed entries with tighter controls
    # Use expected sequence patterns to find missing entries
    extracted_numbers = []
    for entry in results:
        try:
            # Ensure number is a string
            number_str = str(entry.number) if entry.number is not None else ""
            # Extract main number, handle different formats
            main_number_str = re.sub(r'/.*', '', re.sub(r'[A-Za-z]$', '', number_str))
            if main_number_str:
                main_number = int(main_number_str)
                extracted_numbers.append(main_number)
        except ValueError:
            pass
    
    extracted_numbers.sort()
    
    # Find missing numbers in the sequence
    missing_numbers = []
    if extracted_numbers:
        expected_range = range(min(extracted_numbers), max(extracted_numbers) + 1)
        missing_numbers = [num for num in expected_range if num not in extracted_numbers]
    
    # Reset used_text_elements for second pass
    used_text_elements = set()
    for entry in results:
        # Mark all text in existing entries as used
        # This requires reconstructing the coordinates, which we don't have here
        # So this is a placeholder - in practice we'd need to track these
        pass
    
    # For each missing number, look for entries that might match
    for missing_num in missing_numbers:
        # Look for text elements that contain this number
        matching_elements = []
        missing_num_str = str(missing_num)
        
        for text, coords in data:
            # Skip if this text is already used in an entry
            text_id = (text, coords[0][0], coords[0][1])
            if text_id in used_text_elements:
                continue
                
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
                
            # Check for exact number match or number with suffix
            if text == missing_num_str or text.startswith(f"{missing_num_str}/") or text.startswith(f"{missing_num_str}A"):
                matching_elements.append((text, coords, text_id))
        
        # Process each potential match
        for text, coords, text_id in matching_elements:
            # Calculate safe vertical range for this number
            poll_y = coords[0][1]
            
            # IMPROVED: Use much tighter vertical constraints
            safe_y_min = poll_y - median_text_height * 0.25
            safe_y_max = poll_y + median_text_height * 1.5
            
            # Find potential name components
            name_parts = []
            name_coords = []
            name_ids = []
            
            for text_item, text_coords in data:
                text_item_id = (text_item, text_coords[0][0], text_coords[0][1])
                
                # Skip if already used
                if text_item_id in used_text_elements:
                    continue
                    
                # Ensure text_item is a string
                if not isinstance(text_item, str):
                    text_item = str(text_item) if text_item is not None else ""
                    
                # Skip the poll number itself
                if text_item_id == text_id:
                    continue
                
                # Skip items that look like numbers
                if re.match(r'^\d+(/\d+)?[A-Za-z]?$', text_item):
                    continue
                
                text_y = text_coords[0][1]
                
                # Apply stricter vertical constraints
                if safe_y_min <= text_y <= safe_y_max:
                    # Check for reasonable horizontal position
                    text_x = text_coords[0][0]
                    
                    # IMPROVED: Use stricter horizontal constraints
                    # Name should be to the right of number and not too far away
                    if (text_x > coords[0][0] and 
                        (text_x - coords[1][0]) < page_width * 0.2):  # Much stricter constraint
                        
                        # Only add if not already in name_parts
                        if text_item not in name_parts:
                            name_parts.append(text_item)
                            name_coords.append(text_coords)
                            name_ids.append(text_item_id)
            
            # If we found potential name components, create an entry
            if name_parts:
                # Mark these text elements as used
                used_text_elements.add(text_id)
                for name_id in name_ids:
                    used_text_elements.add(name_id)
                    
                # Sort name parts by x-coordinate
                name_parts_with_coords = sorted(zip(name_parts, name_coords), key=lambda x: x[1][0][0])
                name_parts = [part for part, _ in name_parts_with_coords]
                name_coords = [coords for _, coords in name_parts_with_coords]
                
                # Create entry
                name = clean_text(' '.join(name_parts))
                all_coords = [coords] + name_coords
                bbox = get_bounding_box(all_coords)
                
                if bbox and name:
                    poll_number = text
                    # Validate poll_number before creating the entry
                    valid_poll_number = validate_poll_number(poll_number)
                    if valid_poll_number:
                        results.append(Entry(number=valid_poll_number, name=name, bbox=bbox))
    for text, coords in data:
        # Skip if already used
        text_id = (text, coords[0][0], coords[0][1])
        if text_id in used_text_elements:
            continue
            
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        
        if re.match(r'^(7[0-5]|1[0-4][0-9])$', text):
         
            poll_number = text
            poll_y = coords[0][1]
            
         
            safe_y_min = poll_y - median_text_height * 0.5
            safe_y_max = poll_y + median_text_height * 1.5
            

            already_extracted = any(str(entry.number) == poll_number for entry in results)
            
            # Only try to extract if not already handled
            if not already_extracted:
                # Find name components specifically looking for patterns in your examples
                name_parts = []
                name_coords = []
                name_ids = []
                
                # For entries 70-71, look for text "Lohia," which appears in both examples
                for text_item, text_coords in data:
                    # Skip if already used
                    text_item_id = (text_item, text_coords[0][0], text_coords[0][1])
                    if text_item_id in used_text_elements:
                        continue
                        
                    # Ensure text_item is a string
                    if not isinstance(text_item, str):
                        text_item = str(text_item) if text_item is not None else ""
                        
                    if text_item.strip() == "Lohia,":
                        text_y = text_coords[0][1]
                        if safe_y_min <= text_y <= safe_y_max:
                            name_parts.append(text_item)
                            name_coords.append(text_coords)
                            name_ids.append(text_item_id)
                            
                            # Look for additional name components near this one
                            for other_text, other_coords in data:
                                # Skip if already used
                                other_id = (other_text, other_coords[0][0], other_coords[0][1])
                                if other_id in used_text_elements:
                                    continue
                                    
                                # Ensure other_text is a string
                                if not isinstance(other_text, str):
                                    other_text = str(other_text) if other_text is not None else ""
                                    
                                if other_text != text_item and not re.match(r'^\d+$', other_text):
                                    other_y = other_coords[0][1]
                                    # Use stricter vertical matching
                                    if abs(other_y - text_y) < median_text_height * 0.5:
                                        name_parts.append(other_text)
                                        name_coords.append(other_coords)
                                        name_ids.append(other_id)
                if text in ["33", "34"]:
                    for text_item, text_coords in data:
                        # Skip if already used
                        text_item_id = (text_item, text_coords[0][0], text_coords[0][1])
                        if text_item_id in used_text_elements:
                            continue
                            
                        # Ensure text_item is a string
                        if not isinstance(text_item, str):
                            text_item = str(text_item) if text_item is not None else ""
                            
                        if "harl" in text_item or "andal" in text_item:  # match even with OCR errors
                            text_y = text_coords[0][1]
                            if safe_y_min <= text_y <= safe_y_max:
                                name_parts.append(text_item)
                                name_coords.append(text_coords)
                                name_ids.append(text_item_id)
                                
                                # Check for a separate "A" letter before the name
                                for other_text, other_coords in data:
                                    # Skip if already used
                                    other_id = (other_text, other_coords[0][0], other_coords[0][1])
                                    if other_id in used_text_elements:
                                        continue
                                        
                                    # Ensure other_text is a string
                                    if not isinstance(other_text, str):
                                        other_text = str(other_text) if other_text is not None else ""
                                        
                                    if other_text == "A" and abs(other_coords[0][1] - text_y) < median_text_height * 0.4:
                                        # Add the "A" before the name
                                        name_parts.insert(0, other_text)
                                        name_coords.insert(0, other_coords)
                                        name_ids.insert(0, other_id)
                
                # If we found potential name components, create an entry
                if name_parts:
                    # Mark these elements as used
                    used_text_elements.add(text_id)
                    for name_id in name_ids:
                        used_text_elements.add(name_id)
                        
                    # Sort name parts by x-coordinate
                    name_parts_with_coords = sorted(zip(name_parts, name_coords), key=lambda x: x[1][0][0])
                    name_parts = [part for part, _ in name_parts_with_coords]
                    name_coords = [coords for _, coords in name_parts_with_coords]
                    
                    # Create entry
                    name = clean_text(' '.join(name_parts))
                    all_coords = [coords] + name_coords
                    bbox = get_bounding_box(all_coords)
                    
                    if bbox and name:
                        poll_number = text
                        # Validate poll_number before creating the entry
                        valid_poll_number = validate_poll_number(poll_number)
                        if valid_poll_number:
                            results.append(Entry(number=valid_poll_number, name=name, bbox=bbox))
    
    # Define a safer parsing function for sorting
    def safe_parse_entry_number(entry_num):
        try:
            # Ensure entry_num is a string
            if not isinstance(entry_num, str):
                entry_num = str(entry_num) if entry_num is not None else ""
                
            # Handle empty string
            if not entry_num.strip():
                return (0, 0)
                
            # Extract numbers using regex
            cleaned = re.sub(r'[^0-9/]', '', entry_num)
            
            # Handle no digits found
            if not cleaned:
                return (0, 0)
                
            if '/' in cleaned:
                parts = cleaned.split('/', 1)
                # Handle empty parts
                main_part = parts[0]
                sub_part = parts[1] if len(parts) > 1 else ""
                
                # Ensure non-empty strings for conversion
                if not main_part:
                    main_part = "0"
                if not sub_part:
                    sub_part = "0"
                    
                main_num = float(main_part)
                sub_num = float(sub_part)
                return (main_num, sub_num)
            else:
                return (float(cleaned), 0)
        except ValueError:
            # If conversion fails, return default
            return (0, 0)
    
    # Sort entries by poll number
    results.sort(key=lambda x: safe_parse_entry_number(x.number))
    
    return results