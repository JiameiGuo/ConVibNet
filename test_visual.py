import os
import time
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SeqDataset
from model.network_seq import SeqNet
import torch.nn.functional as F
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_model(config, model_path, device):
    model = SeqNet(
        num_angle=config["model"]["num_angle"],
        num_rho=config["model"]["num_rho"],
        seq_len=config["model"]["seq_length"],
        win=config["model"].get("win", 10),
        stride=config["model"].get("stride", 5),
        enc_init=config["model"].get("enc_init", True),
        fic_init=config["model"].get("fic_init", True),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_points(segmentation_mask, threshold=0.5):
    if not isinstance(segmentation_mask, torch.Tensor):
        segmentation_mask = torch.tensor(segmentation_mask)
    # Binarize the mask
    binary_mask = segmentation_mask > threshold
    # Get non-zero coordinates (y, x order in PyTorch)
    coords = torch.nonzero(binary_mask, as_tuple=False)  # Shape: [N, 2]
    coords = coords[:, [1, 0]]
    return coords

def ransac_line_fit(points, max_trials=100, threshold=1.0):
    
    best_inliers = []
    best_line = None
    
    if points.shape[0] < 2:
        return None, []

    for _ in range(max_trials):
        # Randomly sample two points
        sample = points[np.random.choice(points.shape[0], 2, replace=False)]
        x1, y1 = sample[0]
        x2, y2 = sample[1]

        # Calculate the line parameters (m = slope, c = intercept)
        if x1 == x2:  # Prevent division by zero
            continue
        m = (y2 - y1) / float(x2 - x1)
        c = y1 - m * x1

        # Calculate distances of all points to the line
        distances = np.abs(points[:, 1] - (m * points[:, 0] + c)) / np.sqrt(1 + m**2)

        # Identify inliers
        inliers = np.where(distances < threshold)[0]

        # Update the best model if more inliers are found
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (m, c)

    return best_line, best_inliers

def analyze_needle(mask, threshold=0.5, max_trials=100, inlier_threshold=1.0):
    """
    Analyze the needle's position and angle from a binary segmentation mask.
    Args:
        mask: A binary numpy array representing the segmentation mask (after sigmoid).
        threshold: Threshold for binarizing the mask.
        max_trials: Maximum number of iterations for RANSAC.
        inlier_threshold: Distance threshold for RANSAC inliers.
        visualize: Whether to visualize the results.
    Returns:
        needle_tip: The (x, y) coordinates of the needle tip.
        angle_degrees: The angle of the needle in degrees.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a numpy array.")

    coords = extract_points(mask, threshold)
    if coords.numel() == 0:
        return None, None

    points = coords.numpy()
    best_line, best_inliers = ransac_line_fit(points, max_trials, inlier_threshold)

    if best_line is None:
        return None, None

    m, c = best_line
    angle_radians = math.atan(m)
    angle_degrees = math.degrees(angle_radians)

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min = m * x_min + c
    y_max = m * x_max + c

    needle_tip = (x_min, y_min)

    return needle_tip, angle_degrees

def calculate_errors(pred_mask, gt_mask, threshold=0.5, max_trials=100, inlier_threshold=1.0):
    pred_tip, pred_angle = analyze_needle(pred_mask, threshold, max_trials, inlier_threshold)
    gt_tip, gt_angle = analyze_needle(gt_mask, threshold, max_trials, inlier_threshold)
    
    if pred_tip is None or gt_tip is None:
        print("cannot detect needle")
        return None, None

    position_error = np.linalg.norm(np.array(pred_tip) - np.array(gt_tip))

    angle_error = abs(pred_angle - gt_angle)

    return position_error, angle_error

def test(config, model_path, save_dir):
    device = config["train"]["device"]
    os.makedirs(save_dir, exist_ok=True)
    overlay_dir = os.path.join(save_dir, "overlay_images_overfit")
    os.makedirs(overlay_dir, exist_ok=True)
    visual_dir = os.path.join(save_dir, "visual")
    os.makedirs(visual_dir, exist_ok=True) 
    
    model = load_model(config, model_path, device)
    dataset_test = SeqDataset(
        data_path=config["data"]["data_path"],
        split="test_30_15",
        size=config["data"]["size"],
        mask_size=config["data"]["mask_size"],
        seq_length=config["model"]["seq_length"],
        num_angle=config["model"]["num_angle"],
        num_rho=config["model"]["num_rho"],
        augment=False,
    )
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

    tip_errors = []
    angle_errors = []
    num_img = 0
    num_notpred = 0
    total_time = 0.0

    original_height_mm = 45.0
    resized_size = 657 // 2

    pixel_to_mm_scale = original_height_mm / resized_size

    tip_error_threshold_mm = 10
    angle_error_threshold_deg = 15.0

    for i, (img, label) in enumerate(tqdm(test_loader, desc="Testing")):
            
        img = img.to(device)
        label = label.to(device)
        
        start_time = time.time()
        with torch.no_grad():
            pred = model(img)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            # pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
            label = label.unsqueeze(1)
            # pred = F.interpolate(pred, size=label.shape[2:], mode="bilinear", align_corners=False)
            pred = (pred > 0.5).float()
            label = label[0].cpu().numpy().squeeze()  # (H, W)
            pred = pred[0].cpu().numpy().squeeze()  # (H, W)

        if i % 50 == 0:
            img = img[0, -1].cpu().numpy().squeeze()  # (H, W)      
            num_ones_pred = int(pred.sum())
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img, cmap="gray")
            axes[0].set_title("Input Image")
            axes[1].imshow(label, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title("Predicted Mask")

            for ax in axes:
                ax.axis("off")

            plt.suptitle(f"Prediction_{i} (Pixels=1: {num_ones_pred})")
            save_path = os.path.join(visual_dir, f"pred_{i}.png")
            plt.savefig(save_path)
            plt.close()
            
        # cv2.imwrite(f"test_results/pred_{i}.png", (pred * 255).astype(np.uint8))
        # cv2.imwrite(f"test_results/gt_{i}.png", (label * 255).astype(np.uint8))

        pos_error, angle_error = calculate_errors(pred, label)
        num_img += 1
        if pos_error is None or angle_error is None:
            num_notpred += 1
            continue

        pos_error_mm = pos_error * pixel_to_mm_scale

        if pos_error_mm > tip_error_threshold_mm or angle_error > angle_error_threshold_deg:
            num_notpred += 1
            continue

        tip_errors.append(pos_error_mm)
        angle_errors.append(angle_error)

        if i % 50 == 0:
            overlay = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            overlay[label == 1] = [255, 255, 255] 
            overlay[pred == 1] = [0, 0, 255] 

            save_path = os.path.join(overlay_dir, f"overlay_{i:04d}.png")
            cv2.imwrite(save_path, overlay)

    tip_mean = np.mean(tip_errors) if tip_errors else float('nan')
    tip_std = np.std(tip_errors) if tip_errors else float('nan')
    angle_mean = np.mean(angle_errors) if angle_errors else float('nan')
    angle_std = np.std(angle_errors) if angle_errors else float('nan')
    time_avg = total_time / num_img

    print(f"Average Tip Error: {tip_mean:.2f} ± {tip_std:.2f}")
    print(f"Average Angle Error: {angle_mean:.2f} ± {angle_std:.2f}")
    print(f"Predicted Rate: {((num_img - num_notpred) / num_img):.6f}")
    print(f"Average inference time: {time_avg:.4f}")
    print(f"Overlay images saved in: {overlay_dir}")    

if __name__ == "__main__":
    from config import config_list
    
    model_path = "path_to_model/model.pth"
    save_dir = "path_to_dir_for_visualization"
    test(config_list[0], model_path, save_dir)
