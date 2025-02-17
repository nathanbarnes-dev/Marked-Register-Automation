from image_processing import process_pdf
from marked_model.run import run_model
import os
import io
import sys
from contextlib import contextmanager
from image_processing import process_pdf
from marked_model.run import run_model
from PyPDF2 import PdfReader

def delete_files_in_directory(directory_path):
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def multiple_range_handling(csv_path, mode, loading_screen=None):
    delete_files_in_directory("pictures")
    delete_files_in_directory("pictures/duplicates")
    files = []
    
    # Get list of files and calculate total pages
    total_pages = 0
    file_info = []
    
    for root, dirs, filenames in os.walk("pdftorun"):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)
            # Get number of pages in PDF
            with open(file_path, 'rb') as pdf_file:
                pdf = PdfReader(pdf_file)
                num_pages = len(pdf.pages)
                total_pages += num_pages
                file_info.append((file_path, filename, num_pages))
    
    # Initialize loading screen with total pages
    if loading_screen:
        loading_screen.set_total_pages(total_pages)
    
    pages_processed = 0
    for i, (file_path, filename, num_pages) in enumerate(file_info, 1):
        if loading_screen:
            # Update loading screen with file info
            loading_screen.update_queue.put(("file_progress", i, len(files), filename, pages_processed))
            
            # Process with console capture
            with capture_output(loading_screen, pages_processed):
                process_pdf(f"{file_path}")
                poll_prefix = os.path.splitext(filename)[0]
                run_model(poll_prefix, csv_path, mode)
                
            pages_processed += num_pages
        else:
            process_pdf(f"{file_path}")
            poll_prefix = os.path.splitext(filename)[0]
            run_model(poll_prefix, csv_path, mode)
            
        delete_files_in_directory("pictures")
        delete_files_in_directory("pictures/duplicates")
    
    delete_files_in_directory("pdftorun")
    return files

class ConsoleCapture:
    def __init__(self, loading_screen, base_pages):
        self.loading_screen = loading_screen
        self.stdout = sys.stdout
        self.base_pages = base_pages
        
    def write(self, string):
        self.stdout.write(string)  # Still write to console
        # Check if we have a complete line about page processing
        if "Finished processing page" in string:
            try:
                # Extract page number
                page_num = int(string.split("page")[1].strip())
                self.loading_screen.update_queue.put(
                    ("page_progress", self.base_pages + page_num)
                )
            except:
                pass
        self.stdout.flush()
        
    def flush(self):
        self.stdout.flush()

@contextmanager
def capture_output(loading_screen, base_pages):
    console_capture = ConsoleCapture(loading_screen, base_pages)
    old_stdout = sys.stdout
    sys.stdout = console_capture
    try:
        yield console_capture
    finally:
        sys.stdout = old_stdout


def delete_files_in_directory(directory_path):
    """
    Delete all files in the specified directory.
    """
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)



