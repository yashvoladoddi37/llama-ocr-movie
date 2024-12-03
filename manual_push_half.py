from datasets import load_dataset, Dataset #type: ignore
import os
import jsonlines #type: ignore
from PIL import Image

# Paths
posters_dir = "D:\\circuit-house\\llama-ocr-movie\\posters"
jsonl_file_path = "D:\\circuit-house\\llama-ocr-movie\\posters\\zzzzzz.jsonl"

# Print directory contents
print("Directory contents:")
for i, file in enumerate(os.listdir(posters_dir)):
    if i < 5:  # Print first 5 files
        print(f"- {file}")
print("...")

# Get all jpg files
jpg_files = [f for f in os.listdir(posters_dir) if f.endswith('.jpg')]
print(f"\nNumber of JPG files in directory: {len(jpg_files)}")

# Create initial dataset from file paths
initial_dataset = Dataset.from_dict({
    'image_path': [os.path.join(posters_dir, f) for f in jpg_files]
})

# Load images
def load_image(example):
    example['image'] = Image.open(example['image_path'])
    return example

image_dataset = initial_dataset.map(load_image)
print(f"Number of images loaded in dataset: {len(image_dataset)}")

# Load the JSONL data into a mapping dictionary
metadata = {}
with jsonlines.open(jsonl_file_path) as reader:
    for obj in reader:
        metadata[obj['file_name']] = {
            'movie_title': obj['text'],
            'imdb_id': obj['imdb_id']
        }
print(f"Number of entries in JSONL: {len(metadata)}")

def add_metadata(example):
    image_filename = os.path.basename(example['image_path'])
    print(f"Processing: {image_filename}")
    
    # Try to find a match in metadata
    matching_key = next((key for key in metadata if key in image_filename), None)

    if matching_key:
        example['movie_title'] = metadata[matching_key]['movie_title']
        example['imdb_id'] = metadata[matching_key]['imdb_id']
    else:
        print(f"Warning: No metadata found for {image_filename}")
        example['movie_title'] = ""
        example['imdb_id'] = ""
    
    return example

# Apply the mapping function
dataset_with_metadata = image_dataset.map(add_metadata).remove_columns(['image_path'])

reduced_dataset = dataset_with_metadata.select(range(0, len(dataset_with_metadata), 2))  # Take every other row

print(f"Original dataset size: {len(dataset_with_metadata)}")
print(f"Reduced dataset size: {len(reduced_dataset)}")
print("\nSample of reduced dataset:")
print(reduced_dataset[:5])

# Push reduced version to Hugging Face Hub with a different name
reduced_dataset.push_to_hub("yashvoladoddi37/movie-posters-mini-dataset")