import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PyPDF2 import PdfReader, PdfWriter
import os
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import fitz  # PyMuPDF for page preview
from interface import multiple_range_handling

class PageRange:
    def __init__(self):
        self.start = tk.StringVar(value="1")
        self.end = tk.StringVar(value="1")
        self.poll_number = tk.StringVar()

def process_poll_data(file_path, ranges_data):
    """
    Process the poll data from the PDF ranges.
    Creates individual PDFs for each range in a 'pdftorun' folder.
    
    Args:
        file_path (str): Path to the PDF file
        ranges_data (list): List of dictionaries containing range information
            Each dictionary has:
            - start_page: int
            - end_page: int
            - poll_number: str
            
    Returns:
        tuple: (success, message, list of processed poll numbers)
    """
    try:
        # Create pdftorun directory if it doesn't exist
        output_dir = "pdftorun"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Open the source PDF
        reader = PdfReader(file_path)
        
        # Track processed poll numbers
        processed_polls = []
        
        # Process each range
        for range_data in ranges_data:
            # Get range information
            start_page = range_data['start_page'] - 1  # Convert to 0-based index
            end_page = range_data['end_page']
            poll_number = range_data['poll_number']
            
            if not poll_number:  # Skip if no poll number provided
                continue
            
            # Create a new PDF writer
            writer = PdfWriter()
            
            # Add pages for this range
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
            
            # Create output filename
            output_file = os.path.join(output_dir, f"{poll_number}.pdf")
            
            # Save the PDF
            with open(output_file, 'wb') as output:
                writer.write(output)
            
            processed_polls.append(poll_number)
        
        return True, "PDFs created successfully in 'pdftorun' folder", processed_polls
        
    except Exception as e:
        return False, f"Error processing PDFs: {str(e)}", []

class PDFSelector(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("PDF Page Selector")
        self.geometry("1400x1000")
        
        # Variables
        self.pdf_path = tk.StringVar()
        self.csv_path = tk.StringVar()
        self.total_pages = tk.IntVar(value=0)
        self.current_doc = None
        self.page_ranges = []  # List to store PageRange objects
        self.csv_mode = tk.StringVar(value="new")  # "new" or "append"
        
        self.create_widgets()
        
    def create_widgets(self):
        # Configure grid weights for the main window
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Main container
        main_container = ttk.Frame(self)
        main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        # Left panel for PDF selection and ranges
        left_panel = ttk.Frame(main_container, width=450)
        left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 20))
        left_panel.grid_propagate(False)
        
        # Drop zone
        self.drop_frame = ttk.LabelFrame(left_panel, text="Drop PDF Here")
        self.drop_frame.pack(pady=10, fill="x", padx=10)
        
        self.drop_label = ttk.Label(self.drop_frame, text="Drag and drop a PDF file here or click to browse")
        self.drop_label.pack(pady=20, padx=10)
        
        # Configure drop zone
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_label.bind('<Button-1>', self.browse_file)
        
        # File info
        self.file_label = ttk.Label(left_panel, textvariable=self.pdf_path, wraplength=430)
        self.file_label.pack(pady=5, padx=10)
        
        # Total pages label
        self.total_label = ttk.Label(left_panel, text="Total pages: 0")
        self.total_label.pack(pady=5)
        
        # Page ranges frame with scrollbar
        ranges_frame = ttk.LabelFrame(left_panel, text="Page Ranges")
        ranges_frame.pack(pady=10, fill="both", expand=True, padx=10)
        
        # Create a canvas and scrollbar for the ranges
        canvas = tk.Canvas(ranges_frame)
        scrollbar = ttk.Scrollbar(ranges_frame, orient="vertical", command=canvas.yview)
        self.ranges_container = ttk.Frame(canvas)
        
        self.ranges_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.ranges_container, anchor="nw", width=410)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y", pady=5)
        
        # CSV Output section
        csv_frame = ttk.LabelFrame(left_panel, text="CSV Output Settings")
        csv_frame.pack(pady=10, fill="x", padx=10)
        
        # CSV Mode selection
        mode_frame = ttk.Frame(csv_frame)
        mode_frame.pack(fill="x", pady=5)
        
        ttk.Radiobutton(mode_frame, text="Create New CSV", 
                       variable=self.csv_mode, value="new").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Append to Existing", 
                       variable=self.csv_mode, value="append").pack(side="left", padx=5)
        
        # CSV path selection
        path_frame = ttk.Frame(csv_frame)
        path_frame.pack(fill="x", pady=5)
        
        self.csv_path_entry = ttk.Entry(path_frame, textvariable=self.csv_path)
        self.csv_path_entry.pack(side="left", fill="x", expand=True, padx=(5, 2))
        
        ttk.Button(path_frame, text="Browse", command=self.browse_csv).pack(side="left", padx=(2, 5))
        
        # Buttons frame
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(pady=10, padx=10, fill="x")
        
        # Add Range button
        ttk.Button(buttons_frame, text="Add Range", command=self.add_range).pack(side="left", padx=5)
        
        # Process button
        ttk.Button(buttons_frame, text="Process", command=self.process_ranges).pack(side="left", padx=5)
        
        # Error label
        self.error_label = ttk.Label(left_panel, text="", foreground="red", wraplength=430)
        self.error_label.pack(pady=5, padx=10)
        
        # Right panel for previews
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_rowconfigure(3, weight=1)
        
        # Add a title for the preview section
        preview_title = ttk.Label(right_panel, text="Page Previews", font=('Helvetica', 14, 'bold'))
        preview_title.grid(row=0, column=0, pady=(0, 10))
        
        # First page preview
        first_frame = ttk.LabelFrame(right_panel, text="First Page Preview")
        first_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        first_frame.grid_columnconfigure(0, weight=1)
        first_frame.grid_rowconfigure(0, weight=1)
        
        self.first_preview = ttk.Label(first_frame)
        self.first_preview.grid(row=0, column=0, pady=10)
        
        # Separator
        ttk.Separator(right_panel, orient='horizontal').grid(row=2, column=0, sticky="ew", pady=10)
        
        # Last page preview
        last_frame = ttk.LabelFrame(right_panel, text="Last Page Preview")
        last_frame.grid(row=3, column=0, sticky="nsew")
        last_frame.grid_columnconfigure(0, weight=1)
        last_frame.grid_rowconfigure(0, weight=1)
        
        self.last_preview = ttk.Label(last_frame)
        self.last_preview.grid(row=0, column=0, pady=10)
        
        # Add initial range
        self.add_range()
    
    def browse_csv(self):
        if self.csv_mode.get() == "new":
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Select CSV Output Location"
            )
        else:  # append mode
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv")],
                title="Select CSV to Append"
            )
            
        if file_path:
            self.csv_path.set(file_path)
    
    def create_range_frame(self, page_range, index):
        frame = ttk.Frame(self.ranges_container)
        frame.pack(fill="x", pady=5, padx=5)
        
        # Range label
        ttk.Label(frame, text=f"Range {index + 1}:").grid(row=0, column=0, padx=5)
        
        # Start page controls
        start_frame = ttk.Frame(frame)
        start_frame.grid(row=0, column=1, padx=5)
        
        ttk.Button(start_frame, text="-", width=3,
                  command=lambda: self.increment_page(page_range.start, -1)).pack(side='left')
        ttk.Entry(start_frame, textvariable=page_range.start, width=6).pack(side='left', padx=2)
        ttk.Button(start_frame, text="+", width=3,
                  command=lambda: self.increment_page(page_range.start, 1)).pack(side='left')
        
        # To label
        ttk.Label(frame, text="to").grid(row=0, column=2, padx=5)
        
        # End page controls
        end_frame = ttk.Frame(frame)
        end_frame.grid(row=0, column=3, padx=5)
        
        ttk.Button(end_frame, text="-", width=3,
                  command=lambda: self.increment_page(page_range.end, -1)).pack(side='left')
        ttk.Entry(end_frame, textvariable=page_range.end, width=6).pack(side='left', padx=2)
        ttk.Button(end_frame, text="+", width=3,
                  command=lambda: self.increment_page(page_range.end, 1)).pack(side='left')
        
        # Delete button
        delete_frame = ttk.Frame(frame)
        delete_frame.grid(row=0, column=4, padx=10)
        ttk.Button(delete_frame, text="×", width=3, 
                  command=lambda: self.delete_range(frame, page_range)).pack()
        
        # Poll Number entry
        poll_frame = ttk.Frame(frame)
        poll_frame.grid(row=1, column=0, columnspan=5, sticky="ew", pady=5)
        
        ttk.Label(poll_frame, text="Poll Number:").pack(side='left', padx=5)
        ttk.Entry(poll_frame, textvariable=page_range.poll_number, width=45).pack(side='left', padx=5, fill='x', expand=True)
        
        # Add trace for automatic preview updates
        page_range.start.trace_add("write", lambda *args: self.on_page_change(page_range))
        page_range.end.trace_add("write", lambda *args: self.on_page_change(page_range))
        
        return frame
    
    def add_range(self):
        page_range = PageRange()
        if self.total_pages.get() > 0:
            page_range.end.set(str(self.total_pages.get()))
        self.page_ranges.append(page_range)
        self.create_range_frame(page_range, len(self.page_ranges) - 1)
    
    def delete_range(self, frame, page_range):
        if len(self.page_ranges) > 1:
            self.page_ranges.remove(page_range)
            frame.destroy()
            # Recreate all frames to update numbering
            for widget in self.ranges_container.winfo_children():
                widget.destroy()
            for i, pr in enumerate(self.page_ranges):
                self.create_range_frame(pr, i)
        else:
            messagebox.showwarning("Warning", "Cannot delete the last range")
    
    def increment_page(self, var, delta):
        try:
            current = int(var.get())
            new_value = max(1, min(current + delta, self.total_pages.get()))
            var.set(str(new_value))
        except ValueError:
            pass
    
    def on_page_change(self, page_range):
        # Add a small delay to prevent too frequent updates
        self.after(100, lambda: self.delayed_update(page_range))
    
    def delayed_update(self, page_range):
        if self.validate_page_range(page_range, silent=True):
            try:
                start = int(page_range.start.get()) - 1
                end = int(page_range.end.get()) - 1
                self.display_preview(start, self.first_preview)
                self.display_preview(end, self.last_preview)
            except ValueError:
                pass
    
    def handle_drop(self, event):
        file_path = event.data
        file_path = file_path.strip('{}')
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        
        if file_path.lower().endswith('.pdf'):
            self.load_pdf(file_path)
        else:
            messagebox.showerror("Error", "Please drop a PDF file")
    
    def browse_file(self, event=None):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.load_pdf(file_path)
    
    def load_pdf(self, file_path):
        try:
            self.pdf_path.set(file_path)
            self.current_doc = fitz.open(file_path)
            total_pages = len(self.current_doc)
            self.total_pages.set(total_pages)
            self.total_label.config(text=f"Total pages: {total_pages}")
            
            # Update end page for all ranges
            for page_range in self.page_ranges:
                if not page_range.end.get() or int(page_range.end.get()) > total_pages:
                    page_range.end.set(str(total_pages))
            
            # Update preview of the first range
            if self.page_ranges:
                self.delayed_update(self.page_ranges[0])
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading PDF: {str(e)}")
    
    def validate_page_range(self, page_range, silent=False):
        try:
            start = int(page_range.start.get())
            end = int(page_range.end.get())
            
            if start < 1:
                if not silent:
                    self.error_label.config(text="Start page must be at least 1")
                return False
                
            if end > self.total_pages.get():
                if not silent:
                    self.error_label.config(text="End page exceeds total pages")
                return False
                
            if start > end:
                if not silent:
                    self.error_label.config(text="Start page must be less than end page")
                return False
                
            self.error_label.config(text="")
            return True
            
        except ValueError:
            if not silent:
                self.error_label.config(text="Please enter valid page numbers")
            return False
    
    def validate_all_ranges(self):
        for page_range in self.page_ranges:
            if not self.validate_page_range(page_range):
                return False
        return True
    
    def display_preview(self, page_num, label):
        page = self.current_doc[page_num]
        zoom_factor = 1.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        target_height = 800
        target_width = int(target_height / 1.4142)
        
        # Scale to fit A4 proportions
        img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img)
        
        # Update label
        label.configure(image=photo)
        label.image = photo
    
    def process_ranges(self):
        """Process all page ranges and create individual PDFs."""
        if not self.validate_all_ranges():
            return
            
        try:
            if not self.pdf_path.get():
                messagebox.showwarning("Warning", "Please load a PDF first")
                return
                
            if not self.csv_path.get():
                messagebox.showwarning("Warning", "Please select a CSV output location")
                return
            
            # Collect all range data
            ranges_data = []
            for page_range in self.page_ranges:
                if not page_range.poll_number.get().strip():
                    messagebox.showwarning("Warning", "Please enter poll numbers for all ranges")
                    return
                    
                range_data = {
                    'start_page': int(page_range.start.get()),
                    'end_page': int(page_range.end.get()),
                    'poll_number': page_range.poll_number.get().strip()
                }
                ranges_data.append(range_data)
            
            # Call process_poll_data with only the required arguments
            success, message, processed_polls = process_poll_data(
                self.pdf_path.get(), 
                ranges_data
            )
            
            if success:
                messagebox.showinfo("Success", message)
                # Run multiple range handling after successful PDF creation
                multiple_range_handling(csv_path=self.csv_path.get(), mode=self.csv_mode.get())
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error processing ranges: {str(e)}")
    
    def __del__(self):
        if self.current_doc:
            self.current_doc.close()

if __name__ == "__main__":
    app = PDFSelector()
    app.mainloop()