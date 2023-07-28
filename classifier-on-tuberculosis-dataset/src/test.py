import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch.nn.functional as F
import torch.nn as nn

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

# Constants and other configurations.
TEST_DIR = '../input/xray-dataset-processed/test'
BATCH_SIZE = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224
NUM_WORKERS = 4
CLASS_NAMES = ['Normal', 'Tuberculosis']

# Define the model architecture.
class MedicalMNISTCNN(nn.Module):
    def __init__(self, num_classes=None):
        super(MedicalMNISTCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout2d(p=0.4),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.classifier(x)
        return x

def test_transform(IMAGE_RESIZE):
    transform = transforms.Compose([
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
                )
        ])
    return transform

def create_test_set():
    dataset_test = datasets.ImageFolder(
        TEST_DIR, 
        transform=(test_transform(IMAGE_RESIZE))
    )
    return dataset_test

def create_test_loader(dataset_test):
    test_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return test_loader

def save_test_results(tensor, target, output_class, counter):
    """
    This function will save a few test images along with the 
    ground truth label and predicted label annotated on the image.

    :param tensor: The image tensor.
    :param target: The ground truth class number.
    :param output_class: The predicted class number.
    :param counter: The test image number.
    """
    # Move tensor to CPU and denormalize
    image = torch.squeeze(tensor, 0).cpu().numpy()
    image = image / 2 + 0.5
    image = np.transpose(image, (1, 2, 0))
    # Conver to BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt = target.cpu().numpy()
    cv2.putText(
        image, f"GT: {CLASS_NAMES[int(gt)]}", 
        (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (0, 255, 0), 2, cv2.LINE_AA
    )
    cv2.putText(
        image, f"Pred: {CLASS_NAMES[int(output_class)]}", 
        (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (0, 255, 0), 2, cv2.LINE_AA
    )
    cv2.imwrite(f"../outputs/test_image_{counter}.png", image*255.)

def test(model, testloader, DEVICE):
    """
    Function to test the trained model on the test dataset.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.

    Returns:
        predictions_list: List containing all the predicted class numbers.
        ground_truth_list: List containing all the ground truth class numbers.
        acc: The test accuracy.
    """
    model.eval()
    print('Testing model')
    predictions_list = []
    ground_truth_list = []
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass.
            outputs = model(image)
            # Softmax probabilities.
            predictions = F.softmax(outputs,dim=1).cpu().numpy()
            # Predicted class number.
            output_class = np.argmax(predictions)
            # Append the GT and predictions to the respective lists.
            predictions_list.append(output_class)
            ground_truth_list.append(labels.cpu().numpy())
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()

            # Save a few test images.
            if counter % 999 == 0:
                save_test_results(image, labels, output_class, counter)

    acc = 100. * (test_running_correct / len(testloader.dataset))
    return predictions_list, ground_truth_list, acc

if __name__ == '__main__':
    dataset_test = create_test_set()
    test_loader = create_test_loader(dataset_test)

    checkpoint = torch.load('../outputs/model.pth')
    model = MedicalMNISTCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    predictions_list, ground_truth_list, acc = test(
        model, test_loader, DEVICE
    )
    print(f"Test accuracy: {acc:.3f}%")
    # Confusion matrix.
    conf_matrix = confusion_matrix(ground_truth_list, predictions_list)
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        conf_matrix, 
        annot=True,
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES
    )
    plt.savefig('../outputs/heatmap.png')
    plt.close()
