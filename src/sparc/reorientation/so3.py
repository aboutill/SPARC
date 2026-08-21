import logging

import numpy as np


def matrix_log_so3(R):
    """Matrix logarithm of a rotation matrix R: its skew-symmetric
    tangent-space representation at the identity."""
   
    # Clamp trace to [-1, 3] to guard against numerical drift outside [-1, 1]
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # rotation angle in [0, π]

    if theta < 1e-10:
        # Near identity: first-order approximation log(R) ≈ (R - Rᵀ) / 2
        return (R - R.T) / 2.0

    if np.pi - theta < 1e-10:
        # Near π rotation: use the Rodrigues formula for the π case.
        # Find the rotation axis from the symmetric part of R.
        S = (R + np.eye(3)) / 2.0  # columns are candidates for the axis
        # Pick the column with the largest norm for numerical stability
        col = np.argmax(np.linalg.norm(S, axis=0))
        axis = S[:, col]
        axis /= np.linalg.norm(axis)
        # Build the skew-symmetric matrix for axis * π
        return skew(axis * np.pi)

    # General case: Rodrigues' formula
    return (theta / (2.0 * np.sin(theta))) * (R - R.T)


def matrix_exp_so3(omega):
    """Matrix exponential of a skew-symmetric matrix, mapping a
    tangent vector back onto SO(3)."""

    theta = np.sqrt(0.5 * np.sum(omega ** 2))  # ‖ω‖_F / √2 = ‖axis‖ * θ

    if theta < 1e-10:
        return np.eye(3) + omega  # first-order approximation

    return (np.eye(3)
            + (np.sin(theta) / theta) * omega
            + ((1.0 - np.cos(theta)) / theta ** 2) * omega @ omega)


def skew(v):
    """Build the skew-symmetric (cross-product) matrix for vector v."""
    
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def geodesic_distance(R1, R2):
    """Geodesic (Riemannian) distance between two rotation matrices, in radians."""
   
    return np.linalg.norm(matrix_log_so3(R1.T @ R2), "fro") / np.sqrt(2)


def project_to_so3(M):
    """Project an arbitrary 3x3 matrix onto the closest rotation
    matrix in SO(3), via SVD."""
    
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


def chordal_mean(rotations):
    """Chordal (Frobenius) mean of a list of rotation matrices,
    projected back onto SO(3)."""
    
    return project_to_so3(sum(rotations) / len(rotations))


def geodesic_mean_so3(
        rotations,
        max_iter=100,
        tol=1e-10,
    ):
    """Iteratively compute the Riemannian (geodesic/Karcher) mean of
    a list of rotation matrices, initialised from the chordal mean."""
    
    # Initialise
    R = chordal_mean(rotations)
    n = len(rotations)

    grad_norms = []
    dist_history = []

    for iteration in range(max_iter):
        # Step 1 & 2: accumulate weighted tangent vectors
        epsilon_bar = np.zeros((3, 3))
        for Ri in rotations:
            epsilon_bar += matrix_log_so3(R.T @ Ri)
        epsilon_bar /= n

        grad_norm = np.linalg.norm(epsilon_bar, 'fro')
        mean_dist = np.mean([geodesic_distance(R, Ri) for Ri in rotations])

        grad_norms.append(grad_norm)
        dist_history.append(mean_dist)

        # Step 3: update
        R = R @ matrix_exp_so3(epsilon_bar)

        # Keep R numerically on SO(3) (re-project every few steps)
        if (iteration + 1) % 10 == 0:
            R = project_to_so3(R)

        # Step 4: check convergence
        if grad_norm < tol:
            return R

    logging.warning(
        f"geodesic_mean_so3 did not converge after {max_iter} "
        f"iterations (final grad_norm={grad_norm:.2e}, tol={tol:.2e})."
    )
    
    return R