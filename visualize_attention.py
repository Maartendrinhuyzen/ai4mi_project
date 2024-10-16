import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SliceDataset  # Assuming you have a dataset class named SliceDataset
from UnetAttention import UNetAttention  # Assuming UNetAttention is in a file UnetAttention.py
from operator import itemgetter
import numpy as np
from utils import class2one_hot

# Load the trained model with map_location for CPU if GPU is not available
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set model to evaluation mode
    return model


# Visualize and save the attention map for one sample from the dataset
def visualize_sample_and_attention(model, dataset, device):
    # Get a single sample
    sample = dataset[0]  # Assuming you want the first sample (index can be changed)
    image = sample['images'].unsqueeze(0).to(device)  # Add batch dimension and send to device
    original_image = image.squeeze(0).cpu().numpy()  # For plotting the original image
    
    # Run forward pass
    output = model(image)
    
    # Extract attention map from the last attention block
    attention_map = model.up4.attention.attention_map

    # Resize attention map to match the original image size
    attention_map_resized = torch.nn.functional.interpolate(
        attention_map.unsqueeze(0),
        size=(original_image.shape[1], original_image.shape[2]),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    # Normalize attention map for better visualization
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
    
    # Plot the original image and attention overlay
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image[0], cmap='gray')  # Plotting the first channel for grayscale image
    ax[0].set_title("Original Image")

    ax[1].imshow(original_image[0], cmap='gray')
    ax[1].imshow(attention_map_resized.cpu().detach().numpy(), cmap='jet', alpha=0.5)
    ax[1].set_title("Attention Map Overlay")

    # Save the result to a file
    plt.savefig('result_attention.png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    K = 5
    # Load the model
    model_path = 'results/unet_attention_5folds/ce/bestmodel.pkl'
    model = load_model(model_path)
    model.to(device)

    # Dataset part
    B: int = 8
    root_dir = 'data/SEGTHOR_transformed'

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # Normalize to [0, 1]
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        # Mapping classes
        lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add batch dimension
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    dataset = SliceDataset('val',
                            root_dir,
                            img_transform=img_transform,
                            gt_transform=gt_transform,
                            debug=True)

    # Visualize and save the sample and attention
    visualize_sample_and_attention(model, dataset, device)
