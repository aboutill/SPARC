import torch
import torch.nn.functional as F
import numpy as np

from torch import nn


@staticmethod
def image_distance(img_gt, img_pred, beta=0.01, reduction="mean"):
    """Smooth-L1 image loss, normalised by the ground-truth intensity range."""

    img_loss = F.smooth_l1_loss(img_gt, img_pred, beta=beta, reduction=reduction)
    return img_loss / img_gt.max()


@staticmethod
def euclidean_distance(points_gt, points_pred, reduction="mean"):
    """Mean-squared-error loss between predicted and ground-truth
    anchor points, averaged over the three points."""
    
    dist = nn.MSELoss(reduction=reduction)
    
    # Extract points
    points_pred = points_pred.view(-1, 3, 3)
    p1_pred = points_pred[:, 0]
    p2_pred = points_pred[:, 1] 
    p3_pred = points_pred[:, 2]
    
    points_gt = points_gt.view(-1, 3, 3)
    p1_gt = points_gt[:, 0]
    p2_gt = points_gt[:, 1] 
    p3_gt = points_gt[:, 2]
    
    # Point losses
    loss_p1 = dist(p1_gt, p1_pred)
    loss_p2 = dist(p2_gt, p2_pred)
    loss_p3 = dist(p3_gt, p3_pred)
    
    point_loss = (loss_p1 + loss_p2 + loss_p3) / 3
    
    return point_loss


@staticmethod
def geodesic_distance(R_gt, R_pred, eps=1e-7, reduction="mean"):
    """Geodesic (rotation) distance between batches of predicted and
    ground-truth rotation matrices."""
    
    # Rotation distance
    R_diffs = R_gt @ R_pred.permute(0, 2, 1)
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
    
    if reduction == "none":
        return dists
    elif reduction == "mean":
        return dists.mean()
    elif reduction == "sum":
        return dists.sum()
    

@staticmethod
def translation_distance(t_gt, t_pred, reduction="mean"):
    """Mean-squared-error loss between predicted and ground-truth
    translation vectors."""
    
    dist = nn.MSELoss(reduction=reduction)
    loss = dist(t_gt, t_pred)
    return loss


@staticmethod
def peak_signal_to_noise_ratio(img_gt, img_pred, mask=None, reduction="mean"):
    """PSNR between two images, optionally restricted to a mask."""
    
    # Masking
    if mask is not None:
        img_gt = img_gt[mask]
        img_pred = img_pred[mask]
    
    # PSNR
    data_range = img_gt.max() - img_gt.min()
    mse = np.mean((img_gt - img_pred) ** 2)
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr


@staticmethod
def normalised_mutual_information(img_gt, img_pred, mask=None, bins=64):
    """Normalised mutual information between two images via joint
    histogram entropy, optionally restricted to a mask."""
    
    # Masking
    if mask is not None:
        img_gt = img_gt[mask]
        img_pred = img_pred[mask]
    
    # Flatten the images to create 1D arrays
    img_gt_f = img_gt.ravel()
    img_pred_f = img_pred.ravel()

    # Compute the joint histogram
    hist_2d, _, _ = np.histogram2d(img_gt_f, img_pred_f, bins=bins)

    # Compute joint probability distribution
    pxy = hist_2d / np.sum(hist_2d)

    # Compute marginal probabilities
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Compute joint and individual entropies 
    Hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    Hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    Hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))

    # Compute mutual information
    mi = Hx + Hy - Hxy

    # Compute normalized mutual information
    nmi = 2 * mi / (Hx + Hy)

    return nmi