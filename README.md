A practical system for reducing manual annotation effort through automated detection and labeling.
# Bird Detection and Automated Annotation System

This project presents an end-to-end computer vision pipeline for bird detection, automated annotation, and species classification. The system is designed to reduce manual labeling effort and support efficient dataset construction.

---

## Overview

The system integrates three main components:

- **Detection**: Uses pre-trained YOLO models to localize birds in images  
- **Annotation**: Automatically generates Pascal VOC XML labels from detection results  
- **Classification**: Uses a ResNet50-based model trained on a custom dataset (~50 bird species)  

Pipeline:

Detection → Annotation → Classification

---

## Key Features

- Automated annotation generation (Pascal VOC XML)
- Configurable bounding box expansion (10–15%)
- Batch image processing
- GUI tool for running the annotation pipeline
- Top-k prediction output for classification

---

## Example Functionality

- Detect multiple birds in complex scenes  
- Generate structured annotation files automatically  
- Perform fine-grained species classification  

---

## Technical Stack

- Python  
- PyTorch (ResNet50-based classification)  
- YOLO (pre-trained object detection)  
- OpenCV / PIL (image processing)  
- Tkinter (GUI interface)  

---

## Project Structure

    code/
    ├── annotation.py      # YOLO-based detection and XML generation
    ├── classifier.py      # ResNet50-based image classification
    ├── gui.py             # GUI for annotation tool
    ├── class_names.json   # Class label mapping

---

## Notes

- Detection uses pre-trained YOLO models for efficiency  
- Classification model was trained on a curated dataset (~50 species)  
- The system focuses on practical pipeline integration rather than model training from scratch  

---

## Demo

A short demonstration video is included in the application materials.

---

## Author

Haochuan Xie
