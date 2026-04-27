import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

# ==================== 1. Load model and configuration ====================
def load_trained_model(checkpoint_path='checkpoint.pth'):
    """Load trained model and configuration"""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    print(f"   Model loaded: {checkpoint_path}")
    print(f"   Best accuracy: {checkpoint['best_acc']:.4f}")

    # Extract number of classes
    num_classes = None
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    elif 'state_dict' in checkpoint:
        if 'fc.0.weight' in checkpoint['state_dict']:
            num_classes = checkpoint['state_dict']['fc.0.weight'].shape[0]

    print(f"   Number of classes: {num_classes}")

    return checkpoint, num_classes


# ==================== 2. Image preprocessing ====================
def preprocess_image(image_path, img_size=224):
    """
    Preprocess image (same as training pipeline)
    Steps: Resize → CenterCrop → Normalize → Standardize → Reshape
    """

    img = Image.open(image_path).convert('RGB')

    # Resize: keep aspect ratio, shorter side = 256
    width, height = img.size
    if width > height:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Center crop
    new_width, new_height = img.size
    left = (new_width - img_size) / 2
    top = (new_height - img_size) / 2
    right = (new_width + img_size) / 2
    bottom = (new_height + img_size) / 2
    img = img.crop((left, top, right, bottom))

    # Normalize to 0–1
    img_array = np.array(img) / 255.0

    # Standardize using ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # HWC → CHW
    img_array = img_array.transpose((2, 0, 1))

    # Add batch dimension
    tensor = torch.FloatTensor(img_array).unsqueeze(0)

    return tensor, img


def preprocess_image_simple(image_path, img_size=224):
    """Simplified preprocessing using torchvision transforms"""

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()
    tensor = transform(img).unsqueeze(0)

    return tensor, original_img


# ==================== 3. Prediction ====================
def predict_single_image(model, image_tensor, class_names=None, cat_to_name=None, topk=3):
    """
    Predict a single image

    Args:
        model: trained model
        image_tensor: preprocessed image tensor
        class_names: class name list
        cat_to_name: mapping from class to display name
        topk: number of top predictions
    """

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probabilities, topk)

    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    results = []

    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        result = {
            'rank': i + 1,
            'class_index': int(idx),
            'probability': float(prob),
            'confidence': f"{prob * 100:.2f}%"
        }

        # Add class name
        if class_names is not None and idx < len(class_names):
            folder_name = class_names[idx]
            result['folder_name'] = folder_name

            if cat_to_name is not None:
                if isinstance(cat_to_name, dict) and folder_name in cat_to_name:
                    result['chinese_name'] = cat_to_name[folder_name]
                else:
                    result['chinese_name'] = folder_name

        results.append(result)

    return results


# ==================== 4. Visualization ====================
def display_prediction(image_path, original_img, prediction_results, save_path=None):
    """Display image and prediction results"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: input image
    ax1.imshow(original_img)
    ax1.set_title("Input Image", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Right: prediction results
    ax2.axis('off')
    ax2.text(0.1, 0.95, "Model Predictions:", fontsize=16, fontweight='bold',
             transform=ax2.transAxes, verticalalignment='top')

    y_pos = 0.85

    for result in prediction_results:
        rank = result['rank']
        confidence = result['confidence']

        if 'chinese_name' in result:
            display_name = result['chinese_name']
            if 'folder_name' in result:
                display_name += f" ({result['folder_name']})"
        elif 'folder_name' in result:
            display_name = result['folder_name']
        else:
            display_name = f"Class {result['class_index']}"

        color = 'green' if rank == 1 else 'black'
        fontweight = 'bold' if rank == 1 else 'normal'

        text = f"{rank}. {display_name}: {confidence}"
        ax2.text(0.1, y_pos, text, fontsize=13, color=color,
                 fontweight=fontweight, transform=ax2.transAxes)

        y_pos -= 0.08

    filename = os.path.basename(image_path)
    ax2.text(0.1, y_pos, f"File: {filename}", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to: {save_path}")

    plt.show()


# ==================== 5. Main pipeline ====================
def bird_classifier_pipeline():
    """Bird classification pipeline (CLI interface)"""

    print("=" * 60)
    print("🐦 Bird Classification System")
    print("=" * 60)

    checkpoint_path = input("Enter model path (default: checkpoint.pth): ") or "checkpoint.pth"

    try:
        checkpoint, num_classes = load_trained_model(checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("\n Rebuilding model...")
    model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(checkpoint['state_dict'])
    print("Model ready")

    class_names = None
    cat_to_name = None

    if os.path.exists('class_names.json'):
        with open('class_names.json', 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} classes")

    if os.path.exists('cat_to_name.json'):
        with open('cat_to_name.json', 'r', encoding='utf-8') as f:
            cat_to_name = json.load(f)
        print("Loaded name mapping")

    while True:
        print("\nOptions:")
        print("1. Test single image")
        print("2. Test folder")
        print("3. Exit")

        choice = input("Select (1/2/3): ").strip()

        if choice == '3':
            print("Bye!")
            break

        elif choice == '1':
            image_path = input("Enter image path: ").strip()

            if not os.path.exists(image_path):
                print("File not found")
                continue

            print(f"\nProcessing: {os.path.basename(image_path)}")

            try:
                image_tensor, original_img = preprocess_image_simple(image_path)

                results = predict_single_image(
                    model, image_tensor,
                    class_names=class_names,
                    cat_to_name=cat_to_name,
                    topk=5
                )

                display_prediction(image_path, original_img, results)

                print("\nPrediction Results:")
                for result in results:
                    if 'chinese_name' in result:
                        print(f"  {result['rank']}. {result['chinese_name']}: {result['confidence']}")
                    else:
                        print(f"  {result['rank']}. Class {result['class_index']}: {result['confidence']}")

            except Exception as e:
                print(f"Error: {e}")

        elif choice == '2':
            folder_path = input("Enter folder path: ").strip()

            if not os.path.exists(folder_path):
                print("Folder not found")
                continue

            image_files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    image_files.append(os.path.join(folder_path, file))

            if not image_files:
                print("No images found")
                continue

            print(f"Found {len(image_files)} images")

            for i, img_path in enumerate(image_files[:10]):
                try:
                    image_tensor, _ = preprocess_image_simple(img_path)
                    results = predict_single_image(model, image_tensor, topk=1)

                    print(f"{i + 1}. {os.path.basename(img_path)} → {results[0]['confidence']}")

                except Exception as e:
                    print(f"{i + 1}. Error: {e}")

            print("Batch test completed")

        else:
            print("Invalid input")


# ==================== Run ====================
if __name__ == "__main__":
    bird_classifier_pipeline()