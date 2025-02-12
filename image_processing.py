import os
import fitz
import re
from typing import List, Tuple
from text_processing import Entry, ocrtotext, extract_entries, analyze_page_numbers, sort_entries, get_main_number_safe

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    directories = ['pictures', 'pictures/duplicates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def clean_filename(number: str) -> str:
    """Clean the number for use in filenames."""
    cleaned = re.sub(r'[^0-9/]', '', number)
    return cleaned.replace('/', '%')

def crop_pdf_to_image(input_path: str, output_path: str, x1: float, y1: float, x2: float, y2: float, 
                     page_number: int = 0, dpi: int = 300):
    """Crop a section of a PDF page and save it as an image."""
    try:
        doc = fitz.open(input_path)
        page = doc[page_number]
        
        # Convert decimal coordinates to actual points
        page_width = page.rect.width
        page_height = page.rect.height
        actual_coords = (
            x1 * page_width,
            y1 * page_height,
            x2 * page_width,
            y2 * page_height
        )
        
        # Create the crop rectangle and get pixmap
        crop_rect = fitz.Rect(*actual_coords)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, clip=crop_rect)
        
        # Save the image
        pix.save(output_path)
        doc.close()
        
        print(f"Successfully saved cropped image to {output_path}")
        print(f"Converted coordinates: {actual_coords}")
        print(f"Image dimensions: {pix.width}x{pix.height} pixels")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def save_image_with_duplicate_handling(filename: str, output_pdf: str, x1: float, y1: float, 
                                    x2: float, y2: float, page_number: int = 0):
    """Save image with duplicate handling."""
    ensure_directories_exist()
    
    if os.path.exists(output_pdf):
        base_name = os.path.basename(output_pdf)
        duplicate_path = os.path.join('pictures/duplicates', base_name)
        print(f"Duplicate found. Saving to: {duplicate_path}")
        crop_pdf_to_image(filename, duplicate_path, x1, y1, x2, y2, page_number=page_number)
    else:
        crop_pdf_to_image(filename, output_pdf, x1, y1, x2, y2, page_number=page_number)

def process_pdf(filename: str):
    """Process a PDF file to extract entries and save images."""
    # Get OCR results
    raw_text = ocrtotext(filename)
    print(f"Number of pages processed: {len(raw_text)}")
    
    # Process each page
    all_entries = []
    for page_num, page_data in enumerate(raw_text):
        # Extract entries from page
        entries = extract_entries(str(page_data))
        
        # Analyze page numbers for reasonable limits
        page_limits = analyze_page_numbers(entries)
        print(f"Page {page_num + 1} limits determined: up to {page_limits[1]}")
        
        # Filter entries based on reasonable limits
        filtered_entries = [
            entry for entry in entries
            if 0 <= get_main_number_safe(entry.number) <= page_limits[1]
        ]
        
        # Add page numbers to entries
        for entry in filtered_entries:
            entry.page = page_num
            all_entries.append(entry)
    
    # Sort all entries
    sorted_entries = sort_entries(all_entries)
    
    # Save images for each entry
    for entry in sorted_entries:
        number = clean_filename(entry.number)
        output_pdf = f"pictures/{number}.png"
        x1, y1, x2, y2 = entry.bbox
        
        save_image_with_duplicate_handling(
            filename=filename,
            output_pdf=output_pdf,
            x1=x1, y1=y1, x2=x2, y2=y2,
            page_number=entry.page
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python image_processing.py <pdf_file>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    process_pdf(pdf_file)