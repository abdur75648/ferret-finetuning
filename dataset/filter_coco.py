import os
import shutil
import json
from tqdm import tqdm

def filter_images(json_data, source_dir, destination_dir):
    counter = 0
    for item in tqdm(json_data):
        image_filename = item.get("image")
        source_path = os.path.join(source_dir, "COCO_train2014_"+image_filename)
        destination_path = os.path.join(destination_dir, image_filename)

        if os.path.exists(source_path):
            counter += 1
            shutil.copy2(source_path, destination_path)
            # print(f"Image '{image_filename}' copied to 'required_Images'.")
    print(counter, " images filtered!")

if __name__ == "__main__":
    json_file_path = "supgeragi2k_annotations.json"
    images_directory = "train2014"
    required_images_directory = "supgeragi2k_images"
    os.makedirs(required_images_directory, exist_ok=True)
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)
    filter_images(json_data, images_directory, required_images_directory)
