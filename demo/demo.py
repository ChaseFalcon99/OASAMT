import argparse
import os

import cv2
import numpy as np
import scipy.ndimage
import torch

from sam2.build_sam import build_sam2_camera_predictor
from tcn.TCN_Occ_Classifier import TCNOccClassifier
from tcn.TCN_Occ_Predictor import TCNOccPredictor


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Occlusion-aware SAM2 tracking with TCN-based state detection "
        "and trajectory prediction."
    )

    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="../checkpoints/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument(
        "--sam_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        #default="data/video",
        default="data/video",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        #default="data/label",
        default="data/label",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--toc_model",
        type=str,
        default="../tcn/toc_best_model.pt",
    )
    parser.add_argument(
        "--top_model",
        type=str,
        default="../tcn/top_best_model.pt",
    )

    parser.add_argument("--toc_window_size", type=int, default=32)
    parser.add_argument("--top_seq_len", type=int, default=32)
    parser.add_argument("--top_pred_steps", type=int, default=16)
    parser.add_argument("--min_occ_duration", type=int, default=6)
    parser.add_argument("--min_nonocc_duration", type=int, default=0)
    parser.add_argument("--gaussian_sigma_ratio", type=float, default=0.4)
    parser.add_argument("--gap_threshold", type=float, default=10.0)
    parser.add_argument("--display", default=True, action="store_true")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Temporal state smoothing
# -----------------------------------------------------------------------------
class DebouncedSmoother:
    """Debounce binary occlusion predictions across consecutive frames."""

    def __init__(self, min_occ_duration=5, min_nonocc_duration=5):
        self.min_occ_duration = min_occ_duration
        self.min_nonocc_duration = min_nonocc_duration
        self.current_state = 0
        self.buffer_count = 0

    def update(self, pred):
        if pred == self.current_state:
            self.buffer_count = 0
        else:
            self.buffer_count += 1

            if (
                self.current_state == 0
                and pred == 1
                and self.buffer_count >= self.min_occ_duration
            ):
                self.current_state = 1
                self.buffer_count = 0

            elif (
                self.current_state == 1
                and pred == 0
                and self.buffer_count >= self.min_nonocc_duration
            ):
                self.current_state = 0
                self.buffer_count = 0

        return self.current_state


# -----------------------------------------------------------------------------
# Mask and bounding-box utilities
# -----------------------------------------------------------------------------
def generate_color(object_id):
    """Generate the same deterministic BGR color as the original script."""
    np.random.seed(object_id)
    return np.random.randint(0, 255, (3,), dtype=np.uint8)


def show_mask(frame, mask, object_id, id_to_color, borders=True):
    """Overlay a binary mask on a frame."""
    frame_seg = frame.copy()

    if len(mask.shape) == 3:
        mask = np.squeeze(mask)

    mask = mask.astype(bool)
    color = id_to_color.get(object_id, generate_color(object_id))

    mask_colored = np.zeros_like(frame_seg, dtype=np.uint8)
    mask_colored[mask] = color

    frame_seg = cv2.addWeighted(frame_seg, 1.0, mask_colored, 0.5, 0)

    if borders:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(frame_seg, contours, -1, (0, 0, 0), 2)

    return frame_seg


def get_bounding_boxes(mask):
    """Return the bounding box of a binary mask as (x, y, w, h)."""
    mask = np.squeeze(mask)
    y_indices, x_indices = np.where(mask > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    return cv2.boundingRect(np.column_stack((x_indices, y_indices)))


def compute_center(mask):
    """Compute the centroid of a binary mask in (y, x) coordinates."""
    coords = np.argwhere(mask)

    if coords.size == 0:
        return np.array([0, 0])

    return coords.mean(axis=0)


def shift_mask(mask, shift):
    """Shift a binary mask using nearest-neighbor interpolation."""
    shifted = scipy.ndimage.shift(
        mask.astype(float),
        shift=shift,
        order=0,
        mode="constant",
        cval=0.0,
    )
    return (shifted > 0.5).astype(np.uint8)


def compute_center_aligned_iou(mask1, mask2):
    """Compute IoU after centroid alignment."""
    mask1 = np.squeeze(mask1)
    mask2 = np.squeeze(mask2)

    center1 = compute_center(mask1)
    center2 = compute_center(mask2)

    shift_vector = center1 - center2
    mask2_shifted = shift_mask(mask2, shift=shift_vector)

    union_mask = np.logical_or(mask1, mask2_shifted)

    if union_mask.sum() == 0:
        return 0.0

    y_indices, x_indices = np.where(union_mask)

    x1, y1 = x_indices.min(), y_indices.min()
    x2, y2 = x_indices.max() + 1, y_indices.max() + 1

    sub_mask1 = mask1[y1:y2, x1:x2]
    sub_mask2 = mask2_shifted[y1:y2, x1:x2]

    intersection = np.logical_and(sub_mask1, sub_mask2).sum()
    union = np.logical_or(sub_mask1, sub_mask2).sum()

    return intersection / union if union > 0 else 0.0


def compute_area_in_bbox(mask):
    """Compute foreground area inside the mask bounding box."""
    mask = np.squeeze(mask)

    y_indices, x_indices = np.where(mask > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0

    x1, y1 = x_indices.min(), y_indices.min()
    x2, y2 = x_indices.max() + 1, y_indices.max() + 1

    return mask[y1:y2, x1:x2].sum()


def draw_text_with_background(
    img,
    text,
    position,
    font,
    scale,
    text_color,
    bg_color,
    thickness,
):
    """Draw text while preserving the original visualization behavior."""
    cv2.getTextSize(text, font, scale, thickness)

    x, y = position

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        text_color,
        thickness,
    )


def read_init_rect(txt_path):
    """Read the first initialization box from a comma- or tab-separated file."""
    with open(txt_path, "r") as file:
        line = file.readline().strip()

    parts = line.replace("\t", ",").split(",")
    parts = [part for part in parts if part.strip() != ""]

    x, y, w, h = map(float, parts[:4])

    return [x, y, w, h]


def filter_mask(mask, min_area_ratio=0.1, min_abs_area=50):
    """Remove connected components that are too small."""
    mask = np.squeeze(mask)
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()

    filtered_mask = np.zeros_like(mask_uint8, dtype=bool)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= max_area * min_area_ratio and area >= min_abs_area:
            filtered_mask[labels == label_id] = True

    return filtered_mask


def apply_gaussian_window(img, sigma_ratio=0.3, center_xy=None):
    """Apply the original Gaussian spatial window."""
    height, width = img.shape[:2]

    xv, yv = np.meshgrid(
        np.arange(width),
        np.arange(height),
    )

    if center_xy is None:
        center_x, center_y = width / 2, height / 2
    else:
        center_x, center_y = center_xy

    xv_normalized = (xv - center_x) / (width / 2.0)
    yv_normalized = (yv - center_y) / (height / 2.0)

    dist_sq = xv_normalized**2 + yv_normalized**2

    weight = np.exp(
        -dist_sq / (2 * sigma_ratio**2)
    )

    weight = (weight / weight.max())[..., None]

    return (
        img.astype(np.float32) * weight
    ).astype(np.uint8)


def validate_bbox_coords(
    x,
    y,
    w,
    h,
    frame_width,
    frame_height,
):
    """Clamp a bounding box to valid frame coordinates."""
    x = 0.0 if np.isnan(x) or np.isinf(x) else x
    y = 0.0 if np.isnan(y) or np.isinf(y) else y

    w = (
        1.0
        if np.isnan(w) or np.isinf(w) or w <= 0
        else w
    )
    h = (
        1.0
        if np.isnan(h) or np.isinf(h) or h <= 0
        else h
    )

    x = max(0.0, min(x, frame_width - 1))
    y = max(0.0, min(y, frame_height - 1))

    w = max(1.0, min(w, frame_width - x))
    h = max(1.0, min(h, frame_height - y))

    return x, y, w, h


# -----------------------------------------------------------------------------
# Tracking
# -----------------------------------------------------------------------------
def process_video(video_name, args, device):
    """Track one video while preserving the original tracking logic."""

    predictor = build_sam2_camera_predictor(
        args.sam_config,
        args.sam_checkpoint,
        device=device,
    )

    toc_predictor = TCNOccClassifier(
        model_path=args.toc_model,
        window_size=args.toc_window_size,
    )

    smoother = DebouncedSmoother(
        min_occ_duration=args.min_occ_duration,
        min_nonocc_duration=args.min_nonocc_duration,
    )

    top_predictor = TCNOccPredictor(
        model_path=args.top_model,
        seq_len=args.top_seq_len,
        pred_steps=args.top_pred_steps,
    )

    video_path = os.path.join(
        args.video_dir,
        video_name,
    )

    base_name = os.path.splitext(video_name)[0]

    annotation_path = os.path.join(
        args.annotation_dir,
        base_name + ".txt",
    )

    if not os.path.exists(annotation_path):
        print(
            f"Warning: annotation file not found: "
            f"{annotation_path}. Skipping video."
        )
        return

    print(f"Processing {video_name} ...")

    save_dir = os.path.join(
        args.output_dir,
        base_name,
    )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    result_path = os.path.join(
        save_dir,
        f"{base_name}.txt",
    )

    result_file = open(
        result_path,
        "w",
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        result_file.close()
        raise RuntimeError(
            f"Failed to open video: {video_path}"
        )

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    id_to_color = {}

    first_frame = True
    frame_idx = 0
    ann_obj_id = 1

    pre_mask = None
    prev_area = 1.0

    prev_debounced_pred = 0

    last_h_dir = None
    last_v_dir = None

    occluded_once_X = False
    occluded_once_Y = False

    debounced_pred = 0

    x = 0.0
    y = 0.0
    w = 0.0
    h = 0.0

    prev_sam_has_output = False
    prev_top_pred_bbox = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # -----------------------------------------------------------------
            # Gaussian-window logic
            #
            # IMPORTANT:
            # Keep the original condition based on the previous value of
            # `box`. Do not replace it with prev_sam_has_output.
            # -----------------------------------------------------------------
            frame_with_window = frame

            if not first_frame:
                if not box and prev_top_pred_bbox is not None:
                    print(
                        f"[INFO] Frame {frame_idx}: "
                        "Applying Gaussian window at previous TOP center."
                    )

                    top_center_x = (
                        prev_top_pred_bbox[0]
                        + prev_top_pred_bbox[2] / 2.0
                    )

                    top_center_y = (
                        prev_top_pred_bbox[1]
                        + prev_top_pred_bbox[3] / 2.0
                    )

                    frame_with_window = apply_gaussian_window(
                        frame,
                        sigma_ratio=args.gaussian_sigma_ratio,
                        center_xy=(
                            top_center_x,
                            top_center_y,
                        ),
                    )

            # -----------------------------------------------------------------
            # Initialization
            # -----------------------------------------------------------------
            if first_frame:
                x, y, w, h = read_init_rect(
                    annotation_path
                )

                init_bbox = [
                    x,
                    y,
                    w,
                    h,
                ]

                predictor.load_first_frame(
                    frame
                )

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

                box_tensor = np.array(
                    [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    dtype=np.float32,
                )

                (
                    _,
                    out_obj_ids,
                    out_mask_logits,
                ) = predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=ann_obj_id,
                    points=None,
                    labels=None,
                    bbox=box_tensor,
                )

                first_frame = False

                masks = [
                    (
                        out_mask_logits[i] > 0.0
                    ).cpu().numpy()
                    for i in range(
                        len(out_obj_ids)
                    )
                ]

                mask = masks[0]

                frame = show_mask(
                    frame,
                    mask,
                    out_obj_ids[0],
                    id_to_color,
                    borders=True,
                )

                box = get_bounding_boxes(
                    mask
                )

                first_box = box

                (
                    init_x,
                    init_y,
                    init_w,
                    init_h,
                ) = map(
                    float,
                    box,
                )

                top_left = (
                    int(init_x),
                    int(init_y),
                )

                bottom_right = (
                    int(init_x + init_w),
                    int(init_y + init_h),
                )

                pre_mask = np.array(
                    mask
                )

                prev_area = compute_area_in_bbox(
                    pre_mask
                )

                cv2.rectangle(
                    frame,
                    top_left,
                    bottom_right,
                    (255, 0, 0),
                    2,
                )

                first_none_input = 1

                prev_sam_has_output = True
                prev_top_pred_bbox = None

            # -----------------------------------------------------------------
            # Tracking
            # -----------------------------------------------------------------
            else:
                (
                    out_obj_ids,
                    out_mask_logits,
                    score,
                    current_out,
                ) = predictor.track(
                    frame_with_window
                )

                current_sam_has_output = (
                    len(out_obj_ids) > 0
                    and out_mask_logits is not None
                )

                if current_sam_has_output:
                    mask = (
                        out_mask_logits[0] > 0.0
                    ).cpu().numpy()

                else:
                    mask = np.zeros(
                        (
                            frame.shape[0],
                            frame.shape[1],
                        ),
                        dtype=np.uint8,
                    )

                mask = filter_mask(
                    mask,
                    min_area_ratio=0.03,
                    min_abs_area=3,
                )

                frame = show_mask(
                    frame,
                    mask,
                    (
                        out_obj_ids[0]
                        if current_sam_has_output
                        else 0
                    ),
                    id_to_color,
                    borders=True,
                )

                box = get_bounding_boxes(
                    mask
                )

                current_top_pred_bbox = None

                # -------------------------------------------------------------
                # SAM2 has a valid bounding box
                # -------------------------------------------------------------
                if (
                    box is not None
                    and len(box) == 4
                ):
                    x, y, w, h = map(
                        float,
                        box,
                    )

                    x_norm = (
                        x / frame_width
                    )

                    y_norm = (
                        y / frame_height
                    )

                    w_norm = (
                        w / frame_width
                    )

                    h_norm = (
                        h / frame_height
                    )

                    bbox4d = [
                        x_norm,
                        y_norm,
                        w_norm,
                        h_norm,
                    ]

                    top_left = (
                        int(x),
                        int(y),
                    )

                    bottom_right = (
                        int(x + w),
                        int(y + h),
                    )

                    # ---------------------------------------------------------
                    # TCN-based occlusion analysis
                    # ---------------------------------------------------------
                    if score is not None:
                        area = compute_area_in_bbox(
                            mask
                        )

                        area_ratio = (
                            area / prev_area
                            if prev_area > 0
                            else 0
                        )

                        iou = (
                            compute_center_aligned_iou(
                                mask,
                                pre_mask,
                            )
                        )

                        score_value = score.item()

                        toc_predictor.add_features(
                            score_value,
                            iou,
                            area_ratio,
                        )

                        toc_pred = (
                            toc_predictor.predict()
                        )

                        # -----------------------------------------------------
                        # IMPORTANT:
                        # This first state-processing block is intentionally
                        # preserved from the original implementation.
                        # -----------------------------------------------------
                        if toc_pred is not None:
                            debounced_pred = (
                                smoother.update(
                                    toc_pred
                                )
                            )

                            if (
                                prev_debounced_pred == 1
                                and debounced_pred == 0
                            ):
                                print(
                                    f"[INFO] Frame "
                                    f"{frame_idx}: "
                                    "Occlusion -> "
                                    "Non-Occlusion, "
                                    "clearing TOP cache"
                                )

                                top_predictor.clear_cache()

                                top_predictor.add_frame_feature(
                                    bbox4d
                                )

                            if debounced_pred == 0:
                                status = (
                                    "Non-Occlusion"
                                )

                                (
                                    occluded_once_X,
                                    occluded_once_Y,
                                ) = (
                                    False,
                                    False,
                                )

                                (
                                    last_h_dir,
                                    last_v_dir,
                                ) = (
                                    None,
                                    None,
                                )

                                top_predictor.add_frame_feature(
                                    bbox4d
                                )

                                cv2.rectangle(
                                    frame,
                                    top_left,
                                    bottom_right,
                                    (255, 0, 0),
                                    2,
                                )

                                predictor._manage_memory_obj(
                                    frame_idx,
                                    current_out,
                                    is_occluded=False,
                                )

                            elif debounced_pred == 1:
                                status = (
                                    "Occlusion"
                                )

                                predictor._manage_memory_obj(
                                    frame_idx,
                                    current_out,
                                    is_occluded=True,
                                )

                            prev_debounced_pred = (
                                debounced_pred
                            )

                        # -----------------------------------------------------
                        # IMPORTANT:
                        # The original implementation performs another state
                        # block here. It must not be removed or merged.
                        # -----------------------------------------------------
                        if debounced_pred == 0:
                            status = (
                                "Non-Occlusion"
                            )

                            (
                                occluded_once_X,
                                occluded_once_Y,
                            ) = (
                                False,
                                False,
                            )

                            (
                                last_h_dir,
                                last_v_dir,
                            ) = (
                                None,
                                None,
                            )

                            top_predictor.add_frame_feature(
                                bbox4d
                            )

                            cv2.rectangle(
                                frame,
                                top_left,
                                bottom_right,
                                (255, 0, 0),
                                2,
                            )

                            predictor._manage_memory_obj(
                                frame_idx,
                                current_out,
                                is_occluded=False,
                            )

                        elif debounced_pred == 1:
                            status = (
                                "Occlusion"
                            )

                            predictor._manage_memory_obj(
                                frame_idx,
                                current_out,
                                is_occluded=True,
                            )

                            preds = (
                                top_predictor.get_next_cached()
                            )

                            if (
                                preds is not None
                                and len(preds) > 0
                            ):
                                (
                                    x_norm_pred,
                                    y_norm_pred,
                                    w_norm_pred,
                                    h_norm_pred,
                                ) = preds

                                x_pred = (
                                    x_norm_pred
                                    * frame_width
                                )

                                y_pred = (
                                    y_norm_pred
                                    * frame_height
                                )

                                w_pred = (
                                    w_norm_pred
                                    * frame_width
                                )

                                h_pred = (
                                    h_norm_pred
                                    * frame_height
                                )

                                (
                                    x_pred,
                                    y_pred,
                                    w_pred,
                                    h_pred,
                                ) = validate_bbox_coords(
                                    x_pred,
                                    y_pred,
                                    w_pred,
                                    h_pred,
                                    frame_width,
                                    frame_height,
                                )

                                gap_threshold = (
                                    args.gap_threshold
                                )

                                # ---------------------------------------------
                                # Horizontal alignment
                                # ---------------------------------------------
                                if (
                                    not occluded_once_X
                                    and box is not None
                                ):
                                    sam_x1 = x
                                    sam_x2 = x + w

                                    top_x1 = x_pred
                                    top_x2 = (
                                        x_pred
                                        + w_pred
                                    )

                                    gap_left = abs(
                                        top_x1
                                        - sam_x1
                                    )

                                    gap_right = abs(
                                        top_x2
                                        - sam_x2
                                    )

                                    if (
                                        max(
                                            gap_left,
                                            gap_right,
                                        )
                                        > gap_threshold
                                    ):
                                        if (
                                            gap_left
                                            < gap_right
                                        ):
                                            last_h_dir = (
                                                "left"
                                            )

                                            x_pred = (
                                                sam_x1
                                            )

                                        else:
                                            last_h_dir = (
                                                "right"
                                            )

                                            x_pred = (
                                                sam_x2
                                                - w_pred
                                            )

                                        occluded_once_X = (
                                            True
                                        )

                                    else:
                                        last_h_dir = (
                                            None
                                        )

                                elif (
                                    occluded_once_X
                                    and box is not None
                                ):
                                    sam_x1 = x
                                    sam_x2 = x + w

                                    if (
                                        last_h_dir
                                        == "left"
                                    ):
                                        x_pred = (
                                            sam_x1
                                        )

                                    elif (
                                        last_h_dir
                                        == "right"
                                    ):
                                        x_pred = (
                                            sam_x2
                                            - w_pred
                                        )

                                # ---------------------------------------------
                                # Vertical alignment
                                # ---------------------------------------------
                                if (
                                    not occluded_once_Y
                                    and box is not None
                                ):
                                    sam_y1 = y
                                    sam_y2 = y + h

                                    top_y1 = y_pred
                                    top_y2 = (
                                        y_pred
                                        + h_pred
                                    )

                                    gap_top = abs(
                                        top_y1
                                        - sam_y1
                                    )

                                    gap_bottom = abs(
                                        top_y2
                                        - sam_y2
                                    )

                                    if (
                                        max(
                                            gap_top,
                                            gap_bottom,
                                        )
                                        > gap_threshold
                                    ):
                                        if (
                                            gap_top
                                            < gap_bottom
                                        ):
                                            last_v_dir = (
                                                "top"
                                            )

                                            y_pred = (
                                                sam_y1
                                            )

                                        else:
                                            last_v_dir = (
                                                "bottom"
                                            )

                                            y_pred = (
                                                sam_y2
                                                - h_pred
                                            )

                                        occluded_once_Y = (
                                            True
                                        )

                                    else:
                                        last_v_dir = (
                                            None
                                        )

                                elif (
                                    occluded_once_Y
                                    and box is not None
                                ):
                                    sam_y1 = y
                                    sam_y2 = y + h

                                    if (
                                        last_v_dir
                                        == "top"
                                    ):
                                        y_pred = (
                                            sam_y1
                                        )

                                    elif (
                                        last_v_dir
                                        == "bottom"
                                    ):
                                        y_pred = (
                                            sam_y2
                                            - h_pred
                                        )

                                (
                                    x_pred,
                                    y_pred,
                                    w_pred,
                                    h_pred,
                                ) = validate_bbox_coords(
                                    x_pred,
                                    y_pred,
                                    w_pred,
                                    h_pred,
                                    frame_width,
                                    frame_height,
                                )

                                # Keep feeding the corrected TOP box during
                                # occlusion exactly as in the original script.
                                top_predictor.add_frame_feature(
                                    [
                                        x_pred
                                        / frame_width,
                                        y_pred
                                        / frame_height,
                                        w_pred
                                        / frame_width,
                                        h_pred
                                        / frame_height,
                                    ]
                                )

                                cv2.rectangle(
                                    frame,
                                    (
                                        int(x_pred),
                                        int(y_pred),
                                    ),
                                    (
                                        int(
                                            x_pred
                                            + w_pred
                                        ),
                                        int(
                                            y_pred
                                            + h_pred
                                        ),
                                    ),
                                    (0, 0, 255),
                                    2,
                                )

                                current_top_pred_bbox = (
                                    x_pred,
                                    y_pred,
                                    w_pred,
                                    h_pred,
                                )

                        # -----------------------------------------------------
                        # TCN does not yet have enough history
                        # -----------------------------------------------------
                        else:
                            top_predictor.add_frame_feature(
                                bbox4d
                            )

                            cv2.rectangle(
                                frame,
                                top_left,
                                bottom_right,
                                (255, 0, 0),
                                2,
                            )

                            predictor._manage_memory_obj(
                                frame_idx,
                                current_out,
                                is_occluded=False,
                            )

                        pre_mask = mask
                        prev_area = area

                # -------------------------------------------------------------
                # SAM2 does not produce a valid bounding box
                # -------------------------------------------------------------
                else:
                    (
                        occluded_once_X,
                        occluded_once_Y,
                    ) = (
                        False,
                        False,
                    )

                    (
                        last_h_dir,
                        last_v_dir,
                    ) = (
                        None,
                        None,
                    )

                    preds = (
                        top_predictor.get_next_cached()
                    )

                    if (
                        preds is not None
                        and len(preds) > 0
                    ):
                        (
                            x_norm_pred,
                            y_norm_pred,
                            w_norm_pred,
                            h_norm_pred,
                        ) = preds

                        x_pred = (
                            x_norm_pred
                            * frame_width
                        )

                        y_pred = (
                            y_norm_pred
                            * frame_height
                        )

                        w_pred = (
                            w_norm_pred
                            * frame_width
                        )

                        h_pred = (
                            h_norm_pred
                            * frame_height
                        )

                        (
                            x_pred,
                            y_pred,
                            w_pred,
                            h_pred,
                        ) = validate_bbox_coords(
                            x_pred,
                            y_pred,
                            w_pred,
                            h_pred,
                            frame_width,
                            frame_height,
                        )

                        cv2.rectangle(
                            frame,
                            (
                                int(x_pred),
                                int(y_pred),
                            ),
                            (
                                int(
                                    x_pred
                                    + w_pred
                                ),
                                int(
                                    y_pred
                                    + h_pred
                                ),
                            ),
                            (0, 0, 255),
                            2,
                        )

                        x = x_pred
                        y = y_pred
                        w = w_pred
                        h = h_pred

                        current_top_pred_bbox = (
                            x_pred,
                            y_pred,
                            w_pred,
                            h_pred,
                        )

                # Preserve the original state assignments.
                prev_sam_has_output = (
                    current_sam_has_output
                )

                prev_top_pred_bbox = (
                    current_top_pred_bbox
                )

            # -----------------------------------------------------------------
            # Visualization and output
            # -----------------------------------------------------------------
            draw_text_with_background(
                frame,
                f"# {frame_idx:d}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 255, 255),
                (0, 0, 0),
                2,
            )

            save_path = os.path.join(
                save_dir,
                f"{frame_idx:05d}.jpg",
            )

            cv2.imwrite(
                save_path,
                frame,
            )

            if args.display:
                cv2.imshow(
                    "Tracking",
                    frame,
                )

            result_file.write(
                f"{frame_idx},"
                f"{x:.2f},"
                f"{y:.2f},"
                f"{w:.2f},"
                f"{h:.2f}\n"
            )

            frame_idx += 1

            if (
                args.display
                and cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

    finally:
        result_file.close()
        cap.release()

        if args.display:
            cv2.destroyAllWindows()

    print(
        f"Saved results to: "
        f"{result_path}"
    )


def main():
    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    video_list = sorted(
        file_name
        for file_name in os.listdir(
            args.video_dir
        )
        if file_name.lower().endswith(
            ".mp4"
        )
    )

    if not video_list:
        raise ValueError(
            f"No MP4 files found in: "
            f"{args.video_dir}"
        )

    for video_name in video_list:
        process_video(
            video_name,
            args,
            device,
        )


if __name__ == "__main__":
    main()