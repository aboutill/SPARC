import torch
import numpy as np

import torch.nn.functional as F
from scipy.spatial.transform import Rotation

_MASK_FILTER_CACHE = {}


def _build_mask_filters(dilate_range, erode_range):
    key = (tuple(dilate_range), tuple(erode_range))
    if key in _MASK_FILTER_CACHE:
        return _MASK_FILTER_CACHE[key]

    odd = lambda i: i if i % 2 else i + 1
    size_range = sorted(set(range(*erode_range)) | set(range(*dilate_range)))

    kernels = {}
    thresholds = {}
    for j in size_range:
        size = odd(j)
        radius = size // 2
        grid = torch.meshgrid(*[torch.arange(size) for _ in range(3)], indexing="ij")
        squared_distances = torch.stack([(axis - radius) ** 2 for axis in grid], 0).sum(0)
        sphere = (squared_distances <= radius ** 2).float()
        kernels[j] = sphere.view(1, 1, size, size, size)
        thresholds[j] = sphere.sum() / sphere.numel()

    _MASK_FILTER_CACHE[key] = kernels, thresholds
    return kernels, thresholds


def apply_mask(
        self,
        img,
        roi_mask=None,
        prob_roi_mask=0.2,
        prob_bg_mask=0.2,
        prob_dilate=0.8,
        prob_erode=0.8,
        dilate_range=(1,10),
        erode_range=(1,5),
    ):
    """Randomly mask the background or ROI (with random
    dilation/erosion) to augment training input.
    """
    n = img.size(0)

    # Probabilities
    if roi_mask is None:
        masks = ["bg", None]
        probs = [prob_bg_mask, 1 - prob_bg_mask]
    else:
        masks = ["bg", "roi", None]
        probs = [prob_bg_mask, prob_roi_mask, 1 - (prob_bg_mask + prob_roi_mask)]
    mask_choices = np.random.choice(masks, size=n, p=probs)

    # Binary sphere kernels
    kernels, thresholds = _build_mask_filters(dilate_range, erode_range)

    bg_mask = (img.abs() < 1e-6).float()
    mask = torch.ones_like(img)

    plan = []
    for i in range(n):
        if mask_choices[i] == "bg" and np.random.uniform() < prob_erode:
            j = np.random.randint(erode_range[0], erode_range[1])
            plan.append((i, "bg", j if j > 0 else None))
        elif mask_choices[i] == "roi" and np.random.uniform() < prob_dilate:
            j = np.random.randint(dilate_range[0], dilate_range[1])
            plan.append((i, "roi", j if j > 0 else None))

    for i, kind, j in plan:
        if kind == "roi" and j is None:
            mask[i] = roi_mask[i]

    # Group by (kind, kernel size) so each unique kernel is applied to
    # a whole batch of images in one conv3d call
    groups = {}
    for i, kind, j in plan:
        if j is not None:
            groups.setdefault((kind, j), []).append(i)

    for (kind, j), indices in groups.items():
        kernel = kernels[j].to(device=img.device, dtype=img.dtype)
        threshold = thresholds[j]
        src = bg_mask if kind == "bg" else roi_mask
        batch = src[indices]
        counts = F.conv3d(batch, kernel, padding=kernel.shape[-1] // 2)
        # Binarisation by majority of neighbourhood threshold
        frac = counts / kernel.sum()
        binarized = (frac > threshold).float()
        if kind == "bg":
            binarized = 1.0 - binarized
        for local_idx, batch_idx in enumerate(indices):
            mask[batch_idx] = binarized[local_idx]

    if roi_mask is not None:
        mask = torch.maximum(mask, roi_mask)

    return img * mask


def init_affine_matrix(self, n, r_range=None, t_range=None, restricted=False):
    """Sample a random rigid (rotation + translation) affine matrix per batch item."""
    
    # Full angle range by default, no translation
    if r_range is None:
        r_range = [1, 1, 1]
    if t_range is None:
        t_range = [0.0, 0.0, 0.0]
    
    # Euler angles in radians 
    a = 2*np.pi*np.random.rand(n) # 0, 2pi
    b = np.arccos(2 * np.random.rand(n) - 1) # 0, pi
    if restricted:
        c = np.pi * np.random.rand(n) # 0, pi
    else:
        c = np.pi * (2 * np.random.rand(n) - 1) # -pi, pi    
    # Apply range
    a *= r_range[0]
    b *= r_range[1]
    c *= r_range[2]
    r = np.stack([a, b, c], -1)
    
    # Random 
    tx = (np.random.rand(n) - 0.5) 
    ty = (np.random.rand(n) - 0.5)
    tz = (np.random.rand(n) - 0.5)
    
    # Apply range
    tx *= t_range[0]
    ty *= t_range[1]
    tz *= t_range[2]
    t = np.stack([tx,ty,tz], -1)
   
    # Rotation matrix
    R = Rotation.from_euler("ZXZ", r)
    rotvec = R.as_rotvec()
    R = R.as_matrix()
    
    # Affine matrix
    mat = np.zeros((n, 4, 4))
    mat[:,:3,:3] = R
    mat[:,:3,3] = t
    mat[:,3,3] = 1
    mat = torch.from_numpy(mat).to(dtype=torch.float32, device=self.device)
    
    angle = np.linalg.norm(rotvec, axis=1)
    
    return mat, (angle, t)


@staticmethod
def invert_affine_matrix(mat):
    """Invert a batch of rigid affine matrices."""
    
    # Extract rotation matrix and translation vector
    R = mat[:, :3, :3]
    t = mat[:, :3, 3]
    
    # Inverse rotation matrix
    R_inv = R.transpose(-2,-1)
    
    # Inverse affine matrix
    mat_inv = torch.zeros_like(mat)
    mat_inv[:,:3,:3] = R_inv
    mat_inv[:,:3,3] = - torch.matmul(R_inv, t.unsqueeze(-1)).squeeze()
    mat_inv[:,3,3] = 1
    
    return mat_inv


def apply_transform(
        self, 
        img, 
        affine_matrix,
        padding_mode="zeros", 
        mode="bilinear",
    ):
    """Resample a batch of images by a batch of affine matrices,
    cycling through images if batch sizes differ."""
    
    if img is None:
        return None, None
    
    n = img.size(0)
    m = affine_matrix.size(0)

    idx = torch.arange(m, device=img.device) % n
    img_gt = img[idx]

    img_def = self.affine_warp(img_gt, affine_matrix)

    return img_def, img_gt


@staticmethod
def points_to_matrix(points, trans_first=False):
    """Recover a rigid affine matrix from three predicted anchor
    points via Gram-Schmidt orthogonalisation.
    """
    
    # Extract points and vectors
    # p2 is the image/cardiac centre anchor
    # p1, p3 are the left-/right-posterior midpoints
    points = points.view(-1, 3, 3)
    p1 = points[:, 0]
    p2 = points[:, 1] 
    p3 = points[:, 2]
    v1 = p3 - p1
    v2 = p2 - p1

    # Build frame
    nz = torch.cross(v1, v2, -1)
    ny = torch.cross(nz, v1, -1)
    nx = v1

    # Rotation matrix
    R = torch.stack((nx, ny, nz), -1)
    R = R / torch.linalg.norm(R, ord=2, dim=-2, keepdim=True)

    # Translation matrix
    if trans_first:
        t = torch.matmul(R.transpose(-2, -1), p2.unsqueeze(-1)).squeeze()
    else:
        t = p2
    
    # Affine matrix
    mat = torch.zeros((points.size(0), 4, 4), device=points.device)
    mat[:,:3,:3] = R
    mat[:,:3,3] = t
    mat[:,3,3] = 1
    
    return mat


def matrix_to_points(self, mat, trans_first=False):
    """Compute the three canonical anchor points (image centre,
    left-/right-posterior midpoints) transformed by a given affine
    matrix."""
    
    # Image size and resolution
    sx, sy, sz = self.img_size
    
    # Create center and corner points
    p1 = [-(sx - 1) / 2, -(sy - 1) / 2, 0] # (-x, -y, 0)
    p2 = [0, 0, 0] # center is (0,0,0) 
    p3 = [(sx - 1) / 2, -(sy - 1) / 2, 0] # (x, -y, 0)
    
    p1 = torch.tensor(p1, dtype=mat.dtype, device=mat.device) 
    p2 = torch.tensor(p2, dtype=mat.dtype, device=mat.device) 
    p3 = torch.tensor(p3, dtype=mat.dtype, device=mat.device)
    
    p = torch.stack((p1, p2, p3), 0)
    p = p.unsqueeze(0).unsqueeze(-1)
   
    # Rotation matrix and translation vector
    R = mat[:, :3, :3].unsqueeze(1) #
    t = (mat[:, :3, 3].unsqueeze(1)).reshape(-1, 1, 3, 1) 
    
    # Transformed points
    if trans_first:
        p = torch.matmul(R, p + t) 
    else:
        p = torch.matmul(R, p) + t 
    
    return p.view(-1, 9) 