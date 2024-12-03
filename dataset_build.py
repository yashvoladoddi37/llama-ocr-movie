import pandas as pd
import requests
import os
import json

# 1. Read the CSV file
df = pd.read_csv('movie_poster_initial.csv')

# 2. Create posters directory if it doesn't exist
if not os.path.exists('posters_new'):
    os.makedirs('posters_new')

# 3. Prepare JSONL file
jsonl_file_path = 'posters_new/poster_data.jsonl'
with open(jsonl_file_path, 'w') as jsonl_file:

    # 4. Download images and write JSONL entries
    base_url = 'https://image.tmdb.org/t/p/w500'

    for index, row in df.iterrows():
        try:
            # Skip if poster_path is NaN or not a string
            if pd.isna(row['poster_path']):
                print(f"Skipping row {index}: No poster path")
                continue

            # Construct full URL just for downloading
            poster_path = str(row['poster_path']).strip('/')  # Remove any leading/trailing slashes
            image_url = base_url + '/' + poster_path

            # Save image using poster_path as filename
            filename = poster_path.replace('/', '')  # Remove any slashes from filename
            with open(f'posters_new/{filename}', 'wb') as f:
                response = requests.get(image_url)
                if response.status_code == 200:
                    f.write(response.content)
                else:
                    print(f"Failed to download image for row {index}: Status code {response.status_code}")

            # Write JSONL entry
            jsonl_entry = {
                "imdb_id": int(row['tmdb_id']),  # Convert 'tmdb_id' to integer
                "file_name": filename,
                "text": row['title']  # assuming 'title' is the column for movie title
            }
            jsonl_file.write(json.dumps(jsonl_entry) + '\n')

            # Optional: Print progress
            if index % 100 == 0:
                print(f'Downloaded {index} images...')

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue