from image_processing import process_pdf
from marked_model.run import run_model
import os

def delete_files_in_directory(directory_path):
    """
    Delete all files in the specified directory.
    """
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def multiple_range_handling(csv_path,mode):
    delete_files_in_directory("pictures")
    delete_files_in_directory("pictures/duplicates")
    files = []
    # Walk through directory and its subdirectories
    for root, dirs, filenames in os.walk("pdftorun"):
        for filename in filenames:
            # Get the full path by joining the root directory with filename
            file_path = os.path.join(root, filename)
            files.append(file_path)
    for i in files:
        process_pdf(f"{i}")
        filename = os.path.basename(i)  # Gets "something.pdf"
        poll_prefix = os.path.splitext(filename)[0]  # Gets "something"
        run_model(poll_prefix,csv_path,mode)
        delete_files_in_directory("pictures")
        delete_files_in_directory("pictures/duplicates")
    delete_files_in_directory("pdftorun")
    return files


