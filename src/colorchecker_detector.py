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


def choose_json_orientation(
    scene_image: np.ndarray,
    rectified_shape: tuple[int, int, int],
    quadrilateral: np.ndarray,
    relative_patch_centers: np.ndarray,
    measurement_patch_size: tuple[int, int],
    white_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_quad = order_quadrilateral_tl_tr_br_bl(quadrilateral)

    for steps in range(4):
        candidate_quad = rotate_quad_order(ordered_quad, steps)
        homography = compute_rectified_to_scene_homography(rectified_shape, candidate_quad)
        absolute_centers = map_points_homography(relative_patch_centers, homography)
        rgb = retrieve_rgb_from_image(scene_image, absolute_centers, measurement_patch_size)
        if find_white_index(rgb) == white_index:
            return candidate_quad, absolute_centers, rgb

    homography = compute_rectified_to_scene_homography(rectified_shape, ordered_quad)
    absolute_centers = map_points_homography(relative_patch_centers, homography)
    rgb = retrieve_rgb_from_image(scene_image, absolute_centers, measurement_patch_size)
    return ordered_quad, absolute_centers, rgb


def extract_patch_images(
    scene_image: np.ndarray,
    absolute_patch_centers: np.ndarray,
    preview_patch_size: tuple[int, int],
) -> list[np.ndarray]:
    return [extract_patch(scene_image, center, preview_patch_size) for center in absolute_patch_centers]


def reformat_scene_image(image: np.ndarray) -> np.ndarray:
    working_width = SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["working_width"]
    return reformat_image(image, working_width)


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


def detect_and_orient_colorchecker(
    image: np.ndarray,
    *,
    white_index: int,
) -> ColorCheckerDetectionResult:
    scene_image = reformat_scene_image(image)
    detections = detect_colour_checkers_segmentation(image, additional_data=True)
    if not detections:
        raise RuntimeError("No ColorChecker detected in the image.")

    detection = detections[0]
    relative_patch_centers = recover_relative_patch_centers(detection.swatch_masks)
    measurement_patch_size = estimate_measurement_patch_size_from_masks(detection.swatch_masks)
    preview_patch_size = estimate_patch_size_from_centers(
        relative_patch_centers,
        scale=PREVIEW_PATCH_SCALE,
    )
    quadrilateral, absolute_patch_centers, measured_rgb = choose_json_orientation(
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
    )
