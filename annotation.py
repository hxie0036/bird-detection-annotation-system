import os
import cv2
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import re

"""
YOLO-based automatic annotation pipeline.
Converts detection results into Pascal VOC XML format.
"""


# =========================
# 1. Convert to JPG + Rename Images
# =========================

def prepare_images(input_dir):
    folder_name = os.path.basename(input_dir.rstrip("\\/"))
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')

    # Get all valid image files
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    files.sort()

    if not files:
        print("No images found")
        return []

    # === Detect existing numbered images ===
    pattern = re.compile(rf"{folder_name}_(\d+)\.jpg")
    existing_numbers = []

    for f in files:
        match = pattern.match(f)
        if match:
            existing_numbers.append(int(match.group(1)))

    # Start numbering after the largest existing index
    start_index = max(existing_numbers) + 1 if existing_numbers else 1

    new_files = []

    for f in files:
        old_path = os.path.join(input_dir, f)

        # === Skip already correctly named images ===
        if pattern.match(f):
            new_files.append(f)
            continue

        try:
            img = Image.open(old_path).convert("RGB")
        except:
            print(f"Failed to read image: {f}")
            continue

        new_name = f"{folder_name}_{start_index}.jpg"
        new_path = os.path.join(input_dir, new_name)

        # === Prevent overwriting existing files ===
        while os.path.exists(new_path):
            start_index += 1
            new_name = f"{folder_name}_{start_index}.jpg"
            new_path = os.path.join(input_dir, new_name)

        # Save as JPG
        img.save(new_path, "JPEG")

        # === Remove original file if it is not a standard named JPG ===
        if not pattern.match(f):
            try:
                os.remove(old_path)
                print(f"Deleted original file: {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

        new_files.append(new_name)
        start_index += 1

    print("✔ Conversion and incremental renaming completed")
    return sorted(new_files)


# =========================
# 2. Generate XML (Pascal VOC format)
# =========================
def save_xml(filename, width, height, objects, save_path):
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "dataset"
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    # Add detected objects
    for obj in objects:
        obj_item = ET.SubElement(annotation, "object")
        ET.SubElement(obj_item, "name").text = obj["name"]

        bndbox = ET.SubElement(obj_item, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])

    tree = ET.ElementTree(annotation)
    tree.write(save_path)


# =========================
# 3. Main Pipeline (YOLO Auto Annotation)
# =========================
def process_images(input_dir, output_dir, extend_percentage):

    # === Step 1: Prepare images (convert + rename) ===
    image_files = prepare_images(input_dir)
    if not image_files:
        return

    # === Step 2: Load YOLO model ===
    print("Loading YOLO model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

    for image_file in image_files:
        xml_name = os.path.splitext(image_file)[0] + ".xml"
        xml_path = os.path.join(output_dir, xml_name)

        # Skip if already annotated
        if os.path.exists(xml_path):
            print(f"Already exists, skipped: {xml_name}")
            continue

        image_path = os.path.join(input_dir, image_file)

        # Safety check for missing files
        if not os.path.exists(image_path):
            print(f"File not found, skipped: {image_file}")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read: {image_file}")
            continue

        height, width, _ = image.shape
        results = model(image)

        objects = []

        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result

            # Confidence threshold
            if conf < 0.5:
                continue

            class_name = model.names[int(cls)]

            # Only keep "bird" class
            if class_name != "bird":
                continue

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # === Optional bounding box expansion ===
            box_w, box_h = x2 - x1, y2 - y1

            expand_w = int(box_w * extend_percentage / 100)
            expand_h = int(box_h * extend_percentage / 100)

            x1 = max(0, x1 - expand_w)
            y1 = max(0, y1 - expand_h)
            x2 = min(width, x2 + expand_w)
            y2 = min(height, y2 + expand_h)

            objects.append({
                "name": class_name,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2
            })

        # === Save XML annotation ===
        if objects:
            save_xml(image_file, width, height, objects, xml_path)
            print(f"✔ Annotation saved: {xml_name}")