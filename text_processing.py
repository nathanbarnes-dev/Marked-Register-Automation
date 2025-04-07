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
    """Extract entries with better handling of street names and multi-line elements."""
    results = []
    data = eval(poll_data)
    
    # Sort by y-coordinate
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
            # Assuming font height is roughly proportional to the height of the bounding box
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
    
    # Process each line, being careful around potential street names
    for i, line in enumerate(lines):
        entry_groups = group_entries_by_position(line)
        
        for group in entry_groups:
            # Skip groups that should be excluded (like page numbers or street names)
            if should_exclude_group([entry[0] for entry in group]):
                continue
            
            # Extract number and name components
            number = None
            number_coords = None
            name_parts = []
            name_coords = []
            
            # Look for poll number
            for text, coords in group:
                if is_number(text) and not number:  # Take the first valid number
                    number = text
                    number_coords = coords
            
            # If we found a number, collect the rest as name
            if number:
                for text, coords in group:
                    if text != number:  # Anything that's not the number is part of the name
                        name_parts.append(text)
                        name_coords.append(coords)
                
                # Create entry if we have both number and name
                if name_parts:
                    name = clean_text(' '.join(name_parts))
                    if name:
                        all_coords = [number_coords] + name_coords
                        bbox = get_bounding_box(all_coords)
                        if bbox:
                            results.append(Entry(number=number, name=name, bbox=bbox))
    
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