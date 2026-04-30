from dataclasses import dataclass

import cv2
import numpy as np
from colour_checker_detection import detect_colour_checkers_segmentation
from colour_checker_detection import SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC
from colour_checker_detection.detection.segmentation import reformat_image


MEASUREMENT_PATCH_SCALE = 0.95
PREVIEW_PATCH_SCALE = 0.65


@dataclass(frozen=True)
class ColorCheckerDetectionResult:
    scene_image: np.ndarray
    quadrilateral: np.ndarray
    relative_patch_centers: np.ndarray
    absolute_patch_centers: np.ndarray
    measurement_patch_size: tuple[int, int]
    preview_patch_size: tuple[int, int]
    measured_rgb: np.ndarray
    patch_images: list[np.ndarray]
    colour_checker_shape: tuple[int, int, int]
    swatch_masks_shape: tuple[int, int]
    orientation_steps: int
    orientation_mirrored: bool
    detected_white_index: int
    detected_black_index: int


def recover_relative_patch_centers(swatch_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(swatch_masks, dtype=np.float64)
    x = 0.5 * (masks[:, 2] + masks[:, 3])
    y = 0.5 * (masks[:, 0] + masks[:, 1])
    return np.column_stack([x, y])


def estimate_patch_size_from_centers(
    patch_centers: np.ndarray, scale: float
) -> tuple[int, int]:
    centers = patch_centers.reshape(4, 6, 2)
    dx = np.median(np.abs(np.diff(centers[:, :, 0], axis=1)))
    dy = np.median(np.abs(np.diff(centers[:, :, 1], axis=0)))
    patch_w = max(8, int(round(dx * scale)))
    patch_h = max(8, int(round(dy * scale)))
    return patch_w, patch_h


def estimate_measurement_patch_size_from_masks(
    swatch_masks: np.ndarray,
    scale: float = MEASUREMENT_PATCH_SCALE,
) -> tuple[int, int]:
    masks = np.asarray(swatch_masks, dtype=np.float64)
    heights = masks[:, 1] - masks[:, 0]
    widths = masks[:, 3] - masks[:, 2]
    patch_w = max(4, int(round(np.median(widths) * scale)))
    patch_h = max(4, int(round(np.median(heights) * scale)))
    return patch_w, patch_h


def order_quadrilateral_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmin(d)]
    bl = points[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rotate_quad_order(ordered_quad: np.ndarray, steps: int) -> np.ndarray:
    return np.roll(ordered_quad, -steps, axis=0)


def compute_rectified_to_scene_homography(
    rectified_shape: tuple[int, int, int], quadrilateral: np.ndarray
) -> np.ndarray:
    height, width = rectified_shape[:2]
    rectified_corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(rectified_corners, quadrilateral.astype(np.float32))


def map_points_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, homography)
    return mapped.reshape(-1, 2)


def extract_patch(image: np.ndarray, center: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    x, y = [int(round(v)) for v in center]
    half_w = width // 2
    half_h = height // 2
    top = max(0, y - half_h)
    bottom = min(image.shape[0], y + half_h)
    left = max(0, x - half_w)
    right = min(image.shape[1], x + half_w)
    return np.asarray(image[top:bottom, left:right, :3])


def retrieve_rgb_from_image(
    image: np.ndarray, patch_centers: np.ndarray, patch_size: tuple[int, int]
) -> np.ndarray:
    rgb = []
    for center in patch_centers:
        patch_image = extract_patch(image, center, patch_size)
        rgb.append(np.mean(patch_image, axis=(0, 1)))
    return np.asarray(rgb, dtype=np.float64)


def find_white_index(rgb_array: np.ndarray) -> int:
    row_means = rgb_array.mean(axis=1)
    return int(np.argmax(row_means))


def find_black_index(rgb_array: np.ndarray) -> int:
    row_means = rgb_array.mean(axis=1)
    return int(np.argmin(row_means))


def mirror_relative_patch_centers(
    relative_patch_centers: np.ndarray,
    rectified_shape: tuple[int, int, int],
) -> np.ndarray:
    height, width = rectified_shape[:2]
    _ = height
    centers = np.asarray(relative_patch_centers, dtype=np.float64).reshape(4, 6, 2).copy()
    centers[..., 0] = (width - 1.0) - centers[..., 0]
    centers = centers[:, ::-1, :]
    return centers.reshape(-1, 2)


def orientation_score(
    rgb_array: np.ndarray,
    *,
    white_index: int,
    black_index: int,
) -> tuple[float, ...]:
    means = np.asarray(rgb_array, dtype=np.float64).mean(axis=1)
    white_rank = int(np.sum(means > means[white_index]))
    black_rank = int(np.sum(means < means[black_index]))
    neutral_means = means[white_index : black_index + 1]
    neutral_rise = np.maximum(np.diff(neutral_means), 0.0)
    neutral_violation_count = int(np.count_nonzero(neutral_rise > 0.0))
    neutral_violation_sum = float(np.sum(neutral_rise))
    white_black_contrast = float(means[white_index] - means[black_index])
    return (
        float(white_rank),
        float(black_rank),
        float(neutral_violation_count),
        neutral_violation_sum,
        -white_black_contrast,
    )


def choose_json_orientation(
    scene_image: np.ndarray,
    rectified_shape: tuple[int, int, int],
    quadrilateral: np.ndarray,
    relative_patch_centers: np.ndarray,
    measurement_patch_size: tuple[int, int],
    white_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool, int, int]:
    ordered_quad = order_quadrilateral_tl_tr_br_bl(quadrilateral)
    black_index = relative_patch_centers.shape[0] - 1
    best_candidate = None
    best_score = None

    for steps in range(4):
        candidate_quad = rotate_quad_order(ordered_quad, steps)
        homography = compute_rectified_to_scene_homography(rectified_shape, candidate_quad)
        for mirrored in (False, True):
            candidate_centers = (
                mirror_relative_patch_centers(relative_patch_centers, rectified_shape)
                if mirrored
                else relative_patch_centers
            )
            absolute_centers = map_points_homography(candidate_centers, homography)
            rgb = retrieve_rgb_from_image(scene_image, absolute_centers, measurement_patch_size)
            current_white_index = find_white_index(rgb)
            current_black_index = find_black_index(rgb)
            score = orientation_score(rgb, white_index=white_index, black_index=black_index)
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = (
                    candidate_quad,
                    absolute_centers,
                    rgb,
                    steps,
                    mirrored,
                    current_white_index,
                    current_black_index,
                )

    if best_candidate is None:
        homography = compute_rectified_to_scene_homography(rectified_shape, ordered_quad)
        absolute_centers = map_points_homography(relative_patch_centers, homography)
        rgb = retrieve_rgb_from_image(scene_image, absolute_centers, measurement_patch_size)
        return ordered_quad, absolute_centers, rgb, 0, False, find_white_index(rgb), find_black_index(rgb)
    return best_candidate


def extract_patch_images(
    scene_image: np.ndarray,
    absolute_patch_centers: np.ndarray,
    preview_patch_size: tuple[int, int],
) -> list[np.ndarray]:
    return [extract_patch(scene_image, center, preview_patch_size) for center in absolute_patch_centers]


def reformat_scene_image(image: np.ndarray) -> np.ndarray:
    working_width = SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["working_width"]
    return reformat_image(image, working_width)


def _linear_to_display_uint8(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert a linear float image to gamma-corrected uint8 for segmentation.

    Uses the 99th percentile as white point to be robust against hot pixels.
    Applies sRGB transfer function so that the detector sees natural-looking contrast.
    """
    img = np.asarray(linear_rgb, dtype=np.float64)
    img = np.clip(img, 0.0, None)
    p99 = np.percentile(img, 99)
    if p99 > 0:
        img = img / p99
    img = np.clip(img, 0.0, 1.0)
    # sRGB gamma (IEC 61966-2-1)
    img = np.where(
        img <= 0.0031308,
        12.92 * img,
        1.055 * np.power(np.maximum(img, 0.0031308), 1.0 / 2.4) - 0.055,
    )
    return (np.clip(img, 0.0, 1.0) * 255).round().astype(np.uint8)


def _apply_clahe(uint8_rgb: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB to enhance local contrast."""
    lab = cv2.cvtColor(uint8_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def make_scene_preview(
    image: np.ndarray,
    quadrilateral: np.ndarray,
    absolute_patch_centers: np.ndarray,
) -> np.ndarray:
    view = reformat_scene_image(image)
    view = np.asarray(view, dtype=np.float64)
    view = np.clip(view, 0.0, None)
    if np.max(view) > 0:
        view = view / np.max(view)
    view = (view * 255.0).round().astype(np.uint8)
    view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

    quad = np.round(quadrilateral).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(view, [quad], isClosed=True, color=(0, 255, 0), thickness=4)

    for label, center in enumerate(absolute_patch_centers, start=1):
        x, y = [int(round(v)) for v in center]
        colour = (0, 255, 255)
        if label == 15:
            colour = (0, 0, 255)
        elif label == 19:
            colour = (255, 255, 255)
        cv2.circle(view, (x, y), 8, colour, -1)
        cv2.putText(
            view,
            str(label),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colour,
            2,
            cv2.LINE_AA,
        )

    return view


def make_rectified_crop_preview(patch_images: list[np.ndarray]) -> np.ndarray:
    patch_h = 60
    patch_w = 60
    gap = 12
    rows, cols = 4, 6
    height = rows * patch_h + (rows + 1) * gap
    width = cols * patch_w + (cols + 1) * gap
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)

    for label, patch in enumerate(patch_images, start=1):
        patch = np.asarray(patch, dtype=np.float64)
        patch = np.clip(patch, 0.0, None)
        if np.max(patch) > 0:
            patch = patch / np.max(patch)
        patch = (patch * 255.0).round().astype(np.uint8)
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_AREA)

        row = (label - 1) // cols
        col = (label - 1) % cols
        top = gap + row * (patch_h + gap)
        left = gap + col * (patch_w + gap)
        bottom = top + patch_h
        right = left + patch_w
        canvas[top:bottom, left:right] = patch

        colour = (0, 255, 255)
        thickness = 2
        if label == 15:
            colour = (0, 0, 255)
            thickness = 3
        elif label == 19:
            colour = (255, 255, 255)
            thickness = 3
        cv2.rectangle(canvas, (left, top), (right, bottom), colour, thickness)
        cv2.putText(
            canvas,
            str(label),
            (left + 8, top + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )

    return canvas


def show_detection_preview(
    scene_image: np.ndarray,
    quadrilateral: np.ndarray,
    absolute_patch_centers: np.ndarray,
    rectified_patch_images: list[np.ndarray],
    swatch_colours: np.ndarray,
) -> None:
    scene_preview = make_scene_preview(scene_image, quadrilateral, absolute_patch_centers)
    rectified_preview = make_rectified_crop_preview(rectified_patch_images)
    checker_preview = make_json_grid_preview(swatch_colours)

    scene_preview = cv2.resize(scene_preview, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    rectified_preview = cv2.resize(rectified_preview, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    checker_preview = cv2.resize(checker_preview, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)

    cv2.imshow("ColorChecker Detection - Scene", scene_preview)
    cv2.imshow("ColorChecker Detection - Rectified", rectified_preview)
    cv2.imshow("ColorChecker Detection - JSON Grid", checker_preview)
    print("Preview aperta. Premi un tasto in una finestra per chiudere.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_json_grid_preview(swatch_colours: np.ndarray) -> np.ndarray:
    patch_h = 140
    patch_w = 140
    gap = 20
    rows, cols = 4, 6
    height = rows * patch_h + (rows + 1) * gap
    width = cols * patch_w + (cols + 1) * gap
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)

    colours = np.asarray(swatch_colours, dtype=np.float64).reshape(rows, cols, 3)
    colours = np.clip(colours, 0.0, None)
    if np.max(colours) > 0:
        colours = colours / np.max(colours)
    colours = (colours * 255.0).round().astype(np.uint8)
    colours = cv2.cvtColor(colours, cv2.COLOR_RGB2BGR)

    label = 1
    for row in range(rows):
        for col in range(cols):
            top = gap + row * (patch_h + gap)
            left = gap + col * (patch_w + gap)
            bottom = top + patch_h
            right = left + patch_w
            canvas[top:bottom, left:right] = colours[row, col]

            border_colour = (0, 255, 255)
            thickness = 2
            if label == 15:
                border_colour = (0, 0, 255)
                thickness = 4
            elif label == 19:
                border_colour = (255, 255, 255)
                thickness = 4
            cv2.rectangle(canvas, (left, top), (right, bottom), border_colour, thickness)
            cv2.putText(
                canvas,
                str(label),
                (left + 10, top + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                border_colour,
                2,
                cv2.LINE_AA,
            )
            label += 1

    return canvas


_DETECTION_WORKING_WIDTHS = [1440, 960, 720, 480]


def _try_detect(image: np.ndarray) -> tuple[object, np.ndarray] | tuple[None, None]:
    """Try detection across multiple working widths and preprocessing variants.

    For each attempt, scene_image and detection_image share the same resolution
    so returned quadrilateral coordinates are always in scene_image space.
    Tries standard gamma first, then CLAHE-enhanced, at each width.
    Returns (detection, scene_image) on success, (None, None) on failure.
    """
    for width in _DETECTION_WORKING_WIDTHS:
        scene_at_width = reformat_image(image, width)
        for preprocess in (_linear_to_display_uint8, lambda s: _apply_clahe(_linear_to_display_uint8(s))):
            detection_at_width = preprocess(scene_at_width)
            detections = detect_colour_checkers_segmentation(
                detection_at_width, additional_data=True, working_width=width
            )
            if detections:
                return detections[0], scene_at_width
    return None, None


def detect_and_orient_colorchecker(
    image: np.ndarray,
    *,
    white_index: int,
) -> ColorCheckerDetectionResult:
    # The detector receives a gamma-corrected uint8 image at the trial working_width
    # so segmentation finds natural contrast regardless of sensor dynamic range.
    # scene_image is kept linear and at the same resolution as the successful detection,
    # so patch coordinates are always consistent.
    detection, scene_image = _try_detect(image)
    if detection is None:
        raise RuntimeError("No ColorChecker detected in the image.")

    relative_patch_centers = recover_relative_patch_centers(detection.swatch_masks)
    measurement_patch_size = estimate_measurement_patch_size_from_masks(detection.swatch_masks)
    preview_patch_size = estimate_patch_size_from_centers(
        relative_patch_centers,
        scale=PREVIEW_PATCH_SCALE,
    )
    quadrilateral, absolute_patch_centers, measured_rgb, orientation_steps, orientation_mirrored, detected_white_index, detected_black_index = choose_json_orientation(
        scene_image,
        detection.colour_checker.shape,
        detection.quadrilateral,
        relative_patch_centers,
        measurement_patch_size,
        white_index=white_index,
    )
    patch_images = extract_patch_images(scene_image, absolute_patch_centers, preview_patch_size)

    return ColorCheckerDetectionResult(
        scene_image=scene_image,
        quadrilateral=quadrilateral,
        relative_patch_centers=relative_patch_centers,
        absolute_patch_centers=absolute_patch_centers,
        measurement_patch_size=measurement_patch_size,
        preview_patch_size=preview_patch_size,
        measured_rgb=measured_rgb,
        patch_images=patch_images,
        colour_checker_shape=detection.colour_checker.shape,
        swatch_masks_shape=detection.swatch_masks.shape,
        orientation_steps=orientation_steps,
        orientation_mirrored=orientation_mirrored,
        detected_white_index=detected_white_index,
        detected_black_index=detected_black_index,
    )
