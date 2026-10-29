"""Data augmentation implementing the paper's documented operations.
Paper Reference: Section "Data augmentation", Eqs. 3-6 (Pages 7-8).

Operations:
- Translation: delta_x, delta_y in [-5, 15] pixels (Eq. 4)
- Flipping: Horizontal and vertical mirror reflection (Eq. 5)
- Rotation: theta in [-10, +10] degrees (Eq. 6)

Strictly applied to training partitions only to balance under-represented classes.
"""

import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd


def apply_translation(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate image by (dx, dy) matching Eq. 4.
    dx, dy in [-5, 15] pixels.
    """
    h, w = image.shape[:2]
    # M = [[1, 0, dx], [0, 1, dy]]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    return translated


def apply_flip(image: np.ndarray, horizontal: bool = True, vertical: bool = False) -> np.ndarray:
    """Flip image horizontally and/or vertically matching Eq. 5."""
    res = image
    if horizontal and vertical:
        res = cv2.flip(res, -1)
    elif horizontal:
        res = cv2.flip(res, 1)
    elif vertical:
        res = cv2.flip(res, 0)
    return res


def apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle_deg in [-10, +10] matching Eq. 6."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
    rotated = cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated


def transform_image(
    image: np.ndarray,
    dx: int,
    dy: int,
    angle_deg: float,
    h_flip: bool,
    v_flip: bool
) -> np.ndarray:
    """Apply compound transformation adhering strictly to paper bounds."""
    out = image.copy()
    if h_flip or v_flip:
        out = apply_flip(out, horizontal=h_flip, vertical=v_flip)
    if dx != 0 or dy != 0:
        out = apply_translation(out, dx=dx, dy=dy)
    if abs(angle_deg) > 1e-3:
        out = apply_rotation(out, angle_deg=angle_deg)
    return out


def generate_balanced_augmentations(
    train_df: pd.DataFrame,
    target_samples_per_class: Optional[int] = None,
    output_dir: str = "Data/Augmented",
    seed: int = 42,
    save_to_disk: bool = True
) -> pd.DataFrame:
    """Generate deterministic augmented images to balance training classes.
    
    Args:
        train_df: Training manifest dataframe with columns ['resolved_path', 'canonical_label', 'leakage_group_id', etc.]
        target_samples_per_class: Target sample count per class. If None, max class count is used.
        output_dir: Directory where generated augmented images are stored.
        seed: Random seed for deterministic generation.
        save_to_disk: If True, writes JPEG images to output_dir; otherwise generates in-memory records.
        
    Returns:
        DataFrame containing records for the augmented images, including full transformation lineage.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    
    class_counts = train_df["canonical_label"].value_counts().to_dict()
    if target_samples_per_class is None:
        target = max(class_counts.values())
    else:
        target = target_samples_per_class
        
    augmented_records = []
    
    for cls, current_count in sorted(class_counts.items()):
        needed = target - current_count
        if needed <= 0:
            continue
            
        cls_df = train_df[train_df["canonical_label"] == cls].reset_index(drop=True)
        num_source = len(cls_df)
        
        for i in range(needed):
            source_idx = i % num_source
            source_row = cls_df.iloc[source_idx]
            
            # Sample parameters strictly within paper ranges
            angle = float(rng.uniform(-10.0, 10.0))
            dx = int(rng.randint(-5, 16))  # [-5, 15] inclusive
            dy = int(rng.randint(-5, 16))  # [-5, 15] inclusive
            h_flip = bool(rng.choice([True, False]))
            v_flip = bool(rng.choice([True, False]))
            
            aug_filename = f"aug_{cls}_{i:05d}_{source_row['file_name']}"
            aug_resolved_path = os.path.join(output_dir, aug_filename)
            
            if save_to_disk and not os.path.exists(aug_resolved_path):
                # Load source image
                src_img = cv2.imread(source_row["resolved_path"])
                if src_img is None:
                    raise ValueError(f"Could not read source image: {source_row['resolved_path']}")
                aug_img = transform_image(src_img, dx=dx, dy=dy, angle_deg=angle, h_flip=h_flip, v_flip=v_flip)
                cv2.imwrite(aug_resolved_path, aug_img)
                
            record = {
                "file_name": aug_filename,
                "relative_path": os.path.relpath(aug_resolved_path, "Data"),
                "resolved_path": aug_resolved_path,
                "canonical_label": cls,
                "is_augmented": True,
                "source_file": source_row["file_name"],
                "source_path": source_row["resolved_path"],
                "leakage_group_id": source_row["leakage_group_id"],  # Preserves group lineage
                "aug_rotation_deg": angle,
                "aug_dx": dx,
                "aug_dy": dy,
                "aug_h_flip": h_flip,
                "aug_v_flip": v_flip,
                "split": source_row.get("split", "train")
            }
            augmented_records.append(record)
            
    aug_df = pd.DataFrame(augmented_records)
    return aug_df

