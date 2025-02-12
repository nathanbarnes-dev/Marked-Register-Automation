import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import csv
from marked_model.training import MarkingDetector  # Import your model class from training.py

def load_model(model_path, device):
    """
    Load the trained model from a saved state dict
    """
    model = MarkingDetector(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_image(image_path, transform):
    """
    Load and preprocess a single image
    """
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def process_folder(folder_path, model, device, prefix):
    """
    Process all images in a folder and return predictions formatted for CSV output
    """
    # Define the same transform as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    folder_path = Path(folder_path)
    
    # Process each image in the folder
    for img_path in folder_path.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # Prepare image
                image_tensor = process_image(img_path, transform)
                image_tensor = image_tensor.to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    _, predicted = outputs.max(1)
                    
                    # Convert prediction to Y/N format
                    # Y for sidemarked (1) or namemarked (2), N for unmarked (0)
                    is_marked = 'Y' if predicted.item() in [1, 2] else 'N'
                
                
                results.append({
                    'image_name': f"{prefix}{img_path.name}",
                    'marked': is_marked
                })
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
    
    return results

def get_sort_key(filename):
    # Remove '.png' if it exists
    name = filename.replace('.png', '')
    
    # Convert % to / for variant images
    name = name.replace('%', '/')
    
    # Split into main number and variant (if exists)
    parts = name.split('/')
    main_part = parts[0]
    
    # Split the main part into prefix and number
    if '-' in main_part:
        prefix, number_str = main_part.rsplit('-', 1)
        try:
            number = int(number_str)
        except ValueError:
            # Handle cases where the number part might not be a clean integer
            number = int(''.join(filter(str.isdigit, number_str)))
    else:
        # If no hyphen, try to separate alphabetic and numeric parts
        prefix = ''.join(filter(str.isalpha, main_part))
        number = int(''.join(filter(str.isdigit, main_part)))
    
    # If there's a variant number, use it as a secondary sort key
    variant_num = int(parts[1]) if len(parts) > 1 else 0
    
    # Return tuple for sorting (prefix, number, variant_num)
    return (prefix, number, variant_num)

def save_to_csv(results, output_path):
    """
    Save results to a CSV file, appending if the file exists
    """
    # First, read existing entries if file exists
    existing_results = []
    try:
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_results = list(reader)
    except FileNotFoundError:
        existing_results = []

    # Combine existing and new results
    all_results = existing_results + results

    # Remove duplicates based on image_name (keeping the latest entry)
    seen = {}
    unique_results = []
    for result in reversed(all_results):
        image_name = result['image_name']
        if image_name not in seen:
            seen[image_name] = True
            # Remove .png extension if it exists
            if image_name.endswith('.png'):
                result['image_name'] = image_name[:-4]
            # Replace % with /
            result['image_name'] = result['image_name'].replace('%', '/')
            unique_results.append(result)
    
    # Sort all results using the new sorting key
    sorted_items = sorted(unique_results, key=lambda x: get_sort_key(x['image_name']))

    # Write all results back to file
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'marked']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted_items:
            writer.writerow(result)
def main(prefix):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    model_path = 'best_marking_detector.pth'
    test_folder = 'pictures'  # Replace with your test folder path
    output_csv = 'marking_results.csv'  # Output CSV file name
    image_prefix = prefix  # Replace with your desired prefix
    
    # Load model
    model = load_model(model_path, device)
    
    # Process images
    results = process_folder(test_folder, model, device, image_prefix)
    
    # Save results to CSV
    save_to_csv(results, output_csv)

if __name__ == '__main__':
    main()