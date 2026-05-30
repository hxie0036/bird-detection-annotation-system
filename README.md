# Bird Detection and Automated Annotation System

An end-to-end computer vision pipeline for bird detection, automated annotation, and fine-grained species classification across 39 species.

This system reduces manual annotation effort by integrating detection, labeling, and classification into a unified workflow.

---

## Overview

The system integrates three main components:

- **Detection**: Uses pre-trained YOLO models to localize birds in images  
- **Annotation**: Automatically generates Pascal VOC XML labels from detection results  
- **Classification**: Uses a ResNet50-based model trained on a custom dataset (39 bird species)  

**Pipeline:** Detection → Annotation → Classification

This project is developed as an independent research project in collaboration with an external university (which provides the dataset). All algorithm design and implementation are my own work.

---

## Key Features

- Automated annotation generation (Pascal VOC XML)
- Configurable bounding box expansion (10–15%)
- Batch image processing
- GUI tool for running the annotation pipeline
- Top-k prediction output for classification

---

## Methodology Highlights

- **Transfer learning with two-stage fine-tuning**: ResNet50 is first trained with a frozen backbone (feature-extraction stage), then fully unfrozen with a reduced learning rate for fine-tuning, following standard best practice for transfer learning.
- **Data augmentation**: Random rotation, horizontal/vertical flipping, and ImageNet-standard normalisation applied to the training set; validation set kept clean for honest evaluation.
- **Validation-based checkpointing**: The best model is saved based on validation accuracy across training epochs.
- **Achieves 70%+ validation accuracy** on the 39-species fine-grained classification task.

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
    ├── class_names.json   # internal label mapping used by the classifier
    ├── cat_to_name.json   # human-readable labels for UI display
    
---

## Notes

- Detection uses pre-trained YOLO models for efficiency  
- Classification model was trained on a curated dataset (39 species)  
- The system focuses on practical pipeline integration rather than model training from scratch
- Model checkpoint is not included due to size constraints, but can be provided upon request.

---

## Ongoing Work

The system is under active development, with the following directions being explored:
- Extending toward real-time detection and recognition (in progress)
- Improving classification accuracy on visually similar species
- Exploring transferability of the pipeline methodology to other fine-grained recognition domains

---

## Demo

Watch demo: https://drive.google.com/file/d/1HzIcOusc9L2VfTzYGBJyahfCE15Vqog9/view?usp=sharing

---

## Author

Haochuan Xie
