import pandas as pd
import os

# Replace this with the path to your CSV file
file_path = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/american_football/American-Football-Tracking-Testing-15-01-2024-13-55-12.csv"
df = pd.read_csv(file_path)

# Get unique tags
unique_tags = df['Tags'].unique()

for tag in unique_tags:
    # Filter rows by tag
    filtered_df = df[df['Tags'] == tag]
    
    # Split the original file name and extension
    base_name, extension = os.path.splitext(file_path)
    
    # Construct the new file name by adding the tag before the extension
    new_file_name = f"{base_name}_{tag}{extension}"
    
    # Save the filtered data to a new CSV file
    filtered_df.to_csv(new_file_name, index=False)
