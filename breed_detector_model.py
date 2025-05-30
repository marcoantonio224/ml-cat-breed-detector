import torch
import random
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from collections import Counter

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

DATASET_TRAIN_DIR = './cat_dataset/train'
DATASET_TEST_DIR = './cat_dataset/test'
TOTAL_RANDOM_IMAGES = 25

# Load training data
train_dataset = ImageFolder(root=DATASET_TRAIN_DIR, transform=transform)

# Load test data set
test_dataset = ImageFolder(root=DATASET_TEST_DIR, transform=transform)
random_images = random.sample(test_dataset.imgs, TOTAL_RANDOM_IMAGES)

# Load a pre-trained ResNet18 model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Load the cat detecting model into application
model.load_state_dict(torch.load("cat_breed_detector_model.pth"))
model.to(device)
model.eval()


def get_training_data():
    num_of_imgs = Counter(train_dataset.targets)
    breed = train_dataset.classes
    breed_img_count = {breed[i]: num_of_imgs[i] for i in range(len(breed))}
    return breed_img_count


def get_accuracy_and_confidence_list():
    confidences = []
    accuracies = []
    accuracy_count = 0
    for img_path, label in random_images:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, 3)
        top_confidence = top_probs[0][0].item() * 100
        accuracy = 0
        # Check if the breed is in top classes
        # predicted by the modal
        if label in top_classes:
            accuracy = 100
            accuracy_count += 1
        average_accuracy = accuracy_count / TOTAL_RANDOM_IMAGES
        accuracies.append(accuracy)
        confidences.append(top_confidence)
    return accuracies, confidences, average_accuracy


def predict_breed(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    # Apply softmax to get class probabilities
    probabilities = F.softmax(outputs, dim=1)
    top_probs, top_classes = torch.topk(probabilities, 3)
    top_breeds = [train_dataset.classes[idx] for idx in top_classes[0]]
    top_confidences = top_probs[0].detach().cpu().numpy() * 100
    return top_breeds, top_confidences
