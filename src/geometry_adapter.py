"""CPU normalization of the Smart Tool initial figure into a `sly.Bitmap`.

The SDK pinned by `dev_requirements.txt` (and baked into the app image declared
in `config.json`) decodes the downloaded initial figure with
`sly.Bitmap.from_json` only, so polygon and multipolygon figures - including the
concrete geometry of labels that belong to AnyShape classes - can not be used to
initialize the predictor. This module reproduces the same normalization with the
geometry primitives that SDK already ships, so neither an SDK release nor an
image rebuild is needed.

`download_init_mask` is a drop-in replacement for
`supervisely.nn.inference.interactive_segmentation.functional.download_init_mask`:
it keeps the same signature and returns a `sly.Bitmap` in image coordinates.
"""

from numbers import Real
from typing import Any, Dict, List, Tuple

import numpy as np

import supervisely as sly

BITMAP = "bitmap"
POLYGON = "polygon"
MULTIPOLYGON = "multipolygon"
SUPPORTED_GEOMETRY_TYPES = (BITMAP, POLYGON, MULTIPOLYGON)

# Label fields holding the geometry of every supported type. Used to recognize
# legacy payloads that do not carry the "geometryType" field.
_GEOMETRY_FIELDS = ((BITMAP, "bitmap"), (POLYGON, "points"), (MULTIPOLYGON, "parts"))


class InitFigureError(Exception):
    """The initial figure can not be normalized into a mask."""


def download_init_mask(api: sly.Api, figure_id, image_id) -> sly.Bitmap:
    """Download the label of `figure_id` and normalize it into a `sly.Bitmap`.

    The returned bitmap is placed in image coordinates and is guaranteed to fit
    into the image, so it can be expanded into a full-image mask with
    `functional.bitmap_to_mask`.
    """
    if image_id is None:
        raise InitFigureError("Can not initialize a mask: image id is not provided.")
    ann_json = api.annotation.download_json(image_id)
    label = find_label(ann_json, figure_id, image_id)
    height, width = get_image_size(api, ann_json, image_id)
    return label_to_bitmap(label, height, width)


def find_label(ann_json: Dict[str, Any], figure_id, image_id=None) -> Dict[str, Any]:
    """Find the label of `figure_id` in the downloaded annotation."""
    if figure_id is None:
        raise InitFigureError("Can not initialize a mask: figure id is not provided.")
    objects = ann_json.get("objects") if isinstance(ann_json, dict) else None
    if not isinstance(objects, list):
        raise InitFigureError('Downloaded annotation has no "objects" list.')
    labels = [obj for obj in objects if isinstance(obj, dict) and obj.get("id") == figure_id]
    if len(labels) == 0:
        raise InitFigureError(f"Label with id {figure_id} not found in image {image_id}.")
    return labels[0]


def get_image_size(api: sly.Api, ann_json: Dict[str, Any], image_id) -> Tuple[int, int]:
    """Image height and width taken from the annotation, or from the instance."""
    size = ann_json.get("size") if isinstance(ann_json, dict) else None
    if isinstance(size, dict):
        height, width = size.get("height"), size.get("width")
        if _is_positive_int(height) and _is_positive_int(width):
            return int(height), int(width)
    image_info = api.image.get_info_by_id(image_id)
    if image_info is None:
        raise InitFigureError(f"Image {image_id} not found.")
    return image_info.height, image_info.width


def get_geometry_type(label: Dict[str, Any]) -> str:
    """Concrete geometry type of the label (AnyShape classes store it as well)."""
    geometry_type = label.get("geometryType")
    if isinstance(geometry_type, str) and geometry_type != "":
        return geometry_type
    # Legacy payloads may omit the field: recognize the type by the stored geometry.
    stored = [name for name, field in _GEOMETRY_FIELDS if label.get(field) is not None]
    if len(stored) == 1:
        return stored[0]
    raise InitFigureError("Can not detect the geometry type of the initial figure.")


def label_to_bitmap(label: Dict[str, Any], img_height: int, img_width: int) -> sly.Bitmap:
    """Rasterize a supported concrete label geometry into an image-space bitmap."""
    geometry_type = get_geometry_type(label)
    if geometry_type == BITMAP:
        try:
            bitmap = sly.Bitmap.from_json(label)
        except Exception as exc:
            raise InitFigureError(f"Can not read the initial bitmap figure: {exc}")
    elif geometry_type in (POLYGON, MULTIPOLYGON):
        bitmap = _rasterize(_get_parts(label, geometry_type), img_height, img_width)
    else:
        raise InitFigureError(
            f'Geometry "{geometry_type}" can not be used as an initial figure. '
            f"Supported geometries: {', '.join(SUPPORTED_GEOMETRY_TYPES)}."
        )
    return clip_to_image(bitmap, img_height, img_width)


def clip_to_image(bitmap: sly.Bitmap, img_height: int, img_width: int) -> sly.Bitmap:
    """Cut the part of the bitmap that lies outside of the image."""
    top, left = bitmap.origin.row, bitmap.origin.col
    data = bitmap.data
    height, width = data.shape[:2]
    row_from, col_from = max(0, -top), max(0, -left)
    row_to, col_to = min(height, img_height - top), min(width, img_width - left)
    if row_from >= row_to or col_from >= col_to:
        raise InitFigureError("The initial figure lies outside of the image.")
    if (row_from, col_from, row_to, col_to) == (0, 0, height, width):
        return bitmap
    clipped = data[row_from:row_to, col_from:col_to]
    if not clipped.any():
        raise InitFigureError("The initial figure has no pixels inside the image.")
    return sly.Bitmap(clipped, sly.PointLocation(row=top + row_from, col=left + col_from))


def _rasterize(parts: List[Dict[str, Any]], img_height: int, img_width: int) -> sly.Bitmap:
    if not (_is_positive_int(img_height) and _is_positive_int(img_width)):
        raise InitFigureError(f"Invalid image size: {img_height}x{img_width}.")
    mask = np.zeros((img_height, img_width), bool)
    for part in parts:
        # Polygon.draw() fills the exterior and clears the interior contours in its
        # own buffer, so every part is rasterized with its own holes only. Parts are
        # then united: a hole of one part must not erase another part.
        part_mask = np.zeros((img_height, img_width), np.uint8)
        polygon = _to_polygon(part)
        try:
            polygon.draw(part_mask, 1)
        except Exception as exc:
            raise InitFigureError(f"Can not rasterize the initial figure: {exc}")
        np.logical_or(mask, part_mask.astype(bool), out=mask)
    if not mask.any():
        raise InitFigureError("The initial figure is empty or lies outside of the image.")
    return sly.Bitmap(mask)


def _get_parts(label: Dict[str, Any], geometry_type: str) -> List[Dict[str, Any]]:
    if geometry_type == POLYGON:
        points = label.get("points")
        if not isinstance(points, dict):
            raise InitFigureError('Polygon figure has no "points" object.')
        return [points]
    parts = label.get("parts")
    if not isinstance(parts, list) or len(parts) == 0:
        raise InitFigureError('Multipolygon figure has no "parts" list.')
    for part in parts:
        if not isinstance(part, dict):
            raise InitFigureError('Every "parts" element of a multipolygon must be an object.')
    return parts


def _to_polygon(part: Dict[str, Any]) -> sly.Polygon:
    exterior = _ring(part.get("exterior"), "exterior")
    interior = part.get("interior", [])
    if not isinstance(interior, list):
        raise InitFigureError('"interior" must be a list of contours.')
    interior = [_ring(contour, "interior") for contour in interior]
    return sly.Polygon.from_json({"points": {"exterior": exterior, "interior": interior}})


def _ring(points: Any, field: str) -> List[List[int]]:
    if not isinstance(points, list) or len(points) < 3:
        raise InitFigureError(f'"{field}" must be a list of at least 3 points.')
    ring = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise InitFigureError(f'"{field}" point #{index} must be an [x, y] pair.')
        x, y = point
        if not (_is_number(x) and _is_number(y)):
            raise InitFigureError(f'"{field}" point #{index} must be a pair of numbers.')
        ring.append([int(round(float(x))), int(round(float(y)))])
    return ring


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
