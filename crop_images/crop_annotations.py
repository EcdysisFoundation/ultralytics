import json
import os
from PIL import Image


def crop_and_save_images(image_path, bounding_boxes, output_dir="cropped_images"):
    """
    Crops images based on a list of bounding boxes and saves them to a directory.

    Args:
        image_path (str): The path to the source image.
        bounding_boxes (list of tuples): A list of bounding box coordinates.
                                         Each tuple should be in the format (x_min, y_min, x_max, y_max).
        output_dir (str): The directory where the cropped images will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # avoid DecompressionBombError
    max_image_pixels = Image.MAX_IMAGE_PIXELS
    print(f'MAX_IMAGE_PIXES is {Image.MAX_IMAGE_PIXELS}')
    if max_image_pixels < 180000000:
        Image.MAX_IMAGE_PIXELS = max_image_pixels * 4
        print(f'raised MAX_IMAGE_PIXES to {Image.MAX_IMAGE_PIXELS}')

    print(f'{len(bounding_boxes)} annotations to save')

    # Open the source image using PIL for easy cropping
    try:
        source_image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The image at '{image_path}' was not found.")
        return

    # Extract the base filename for saving
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    for i, bbox in enumerate(bounding_boxes):
        try:
            x_min, y_min, x_max, y_max = map(int, bbox)

            # Ensure coordinates are within image bounds
            img_width, img_height = source_image.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)

            # Crop the image
            cropped_image = source_image.crop((x_min, y_min, x_max, y_max))

            # Save the cropped image
            output_filename = f"{base_filename}_crop_{i}.png"
            cropped_image.save(os.path.join(output_dir, output_filename))
            print(f"Saved cropped image: {output_filename}")

        except (ValueError, IndexError) as e:
            print(f"Skipping invalid bounding box {bbox}: {e}")


def convert_coco_bbox_to_pil(bbox):
    # convert coco formatted bounding box to PIL format
    x, y, width, height = bbox
    x_min = x
    y_min = y
    x_max = x + width
    y_max = y + height
    return (x_min, y_min, x_max, y_max)


def convert_ls_to_coco_to_pil(bbox, image_width, image_height):
    # undo format_result_label_studio() formatted boundingbox
    x, y, width, height = bbox
    x = x / 100 * image_width
    y = y / 100 * image_height
    width = width / 100 * image_width
    height = height / 100 * image_height
    return convert_coco_bbox_to_pil((x, y, width, height))


# Example usage:
if __name__ == "__main__":
    """
    Use annotation data, or prediction data, as formatted by format_result_label_studio()
    """
    # Example 1: Use static coordinates (for annotated data)
    source_image_file = "source_images/134854c0-f889-4933-9139-3d77f201be85_panorama__1.jpg"
    source_json_file = 'source_json/134854c0-f889-4933-9139-3d77f201be85.json'

    try:
        with open(source_json_file, 'r') as f:
            source_json = json.load(f)
            static_bboxes = [
                convert_ls_to_coco_to_pil(
                    (v['x'], v['y'], v['width'], v['height']),
                    v['original_width'],
                    v['original_height']) for v in source_json
                ]
    except FileNotFoundError:
        print(f"Error: {source_json_file} not found. Please ensure the file exists.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {source_json_file}.")
    # Replace "path/to/your/image.jpg" with the actual path to your image.
    # The script will create a folder named `cropped_images` in the same directory.
    crop_and_save_images(source_image_file, static_bboxes)
