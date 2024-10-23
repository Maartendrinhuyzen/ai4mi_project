import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SliceDataset
from utils import collect_patient_slices
from operator import itemgetter
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images,
                   average_hausdorff_distance,
                   intersection,
                   union,
                   torch2D_Hausdorff_distance, 
                   collect_patient_slices,
                   calculate_3d_dice,
                   calculate_3d_iou,
                   calculate_3d_hausdorff)

def load_saved_model(model_path):
    """
    Loads the saved PyTorch model.

    Parameters:
    - model_path (str): Path to the saved model file.

    Returns:
    - model (torch.nn.Module): The loaded model in evaluation mode.
    """
    # Load the entire model using torch.load
    model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model

def validate3hd_main():
    """
    Main function to load ground truths, generate predictions, and collect them using collect_patient_slices.
    """
    # Configuration
    model_path = "/home/scur2459/ai4mi_project/results/UnetA/combined/bestmodel.pkl"  # Path to the best model
    assert Path(model_path).is_file(), f"Model file not found at {model_path}"
    model = load_saved_model(model_path)

    dataset_name = "SEGTHOR_transformed"  # Dataset name
    data_dir = "/home/scur2459/ai4mi_project/data"  # Data directory
    K = 5  # Number of classes, adjust based on your dataset
    batch_size = 8
    num_workers = 5
    debug = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define transformations for images and ground truths
    img_transform = transforms.Compose([
        lambda img: img.convert('L'),  # Convert to grayscale
        lambda img: np.array(img)[np.newaxis, ...],  # Add channel dimension
        lambda nd: nd / 255.0,  # Normalize to [0, 1]
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],  # Convert to NumPy array
        lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # Normalize class labels
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add channel dimension
        lambda t: torch.nn.functional.one_hot(t, num_classes=K),  # One-hot encode
        lambda t: t.squeeze(0).permute(2, 0, 1).float()  # Rearrange dimensions [C, H, W]
    ])

    # Initialize the validation dataset with both images and ground truths
    root_dir = Path(data_dir) / dataset_name
    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=gt_transform,
                           debug=debug)

    # Initialize DataLoader for validation
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    # Initialize patient_slices_val dictionary
    patient_slices_val = {}
    log_3d_dice_val = {}
    log_3d_iou_val = {}
    print(">> Loading Ground Truths and Generating Predictions...")
    model.to(device)
    model.eval()
    log_3d_ahd_val = {}
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing Batches"):
            images = batch['images'].to(device)  # Shape: [B, C, H, W]
            gt = batch['gts'].to(device)         # Shape: [B, K, H, W]
            img_paths = [image_path.split('_')[1] for image_path in batch['stems']]              # Assuming 'stems' contains patient identifiers

            # Ensure the data range is valid
            assert 0 <= images.min() and images.max() <= 1, "Image data out of range [0, 1]"

            B, _, H, W = images.shape

            # Forward pass to get predictions
            pred_logits = model(images)
            pred_probs = F.softmax(1 * pred_logits, dim=1)  # Softmax across classes

            # For each sample in the batch
            pred_seg = probs2one_hot(pred_probs)
            
            patient_slices_val = collect_patient_slices(patient_slices_val, img_paths, pred_seg, gt, B)
    log_3d_dice_val[21] = calculate_3d_dice(patient_slices_val)
    log_3d_ahd_val[21] = calculate_3d_hausdorff(patient_slices_val)   
    log_3d_iou_val[21] = calculate_3d_iou(patient_slices_val)
    print(log_3d_dice_val)
    print(log_3d_ahd_val)

    # Calculate mean HD and HD95 scores for each epoch
    mean_hd_scores = []
    mean_hd95_scores = []

    for epoch in log_3d_ahd_val:
        hd_scores = []
        hd95_scores = []
        
        for patient_id, metrics in log_3d_ahd_val[epoch].items():
            hd_scores.extend(metrics['HD'])
            hd95_scores.extend(metrics['HD95'])
    
        mean_hd_scores.append(np.mean(hd_scores))
        mean_hd95_scores.append(np.mean(hd95_scores))

    print("Mean HD scores per epoch:", mean_hd_scores)
    print("Mean HD95 scores per epoch:", mean_hd95_scores)
    # epochs = list(log_3d_ahd_val.keys())
    # ahd_scores = [np.mean(list(log_3d_ahd_val[e].values())) for e in epochs]
    # print(ahd_scores)

    epochs = list(log_3d_dice_val.keys())
    dice_scores = [np.mean(list(log_3d_dice_val[e].values())) for e in epochs]
    print(dice_scores)

    epochs = list(log_3d_iou_val.keys())
    iou_scores = [np.mean(list(log_3d_iou_val[e].values())) for e in epochs]
    print(iou_scores)

    # with open('log_3d_ahd_val.pkl', 'wb') as f:
    #     pickle.dump(log_3d_ahd_val, f)
if __name__ == "__main__":
    validate3hd_main()