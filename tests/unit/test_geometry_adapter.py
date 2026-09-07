"""Offline CPU tests for the initial figure geometry normalization.

Rasterization is executed for real; only the instance API is replaced by a stub.
No weights, model or GPU are involved.
"""

import numpy as np
import pytest

import supervisely as sly
from supervisely.nn.inference.interactive_segmentation import functional

from src import geometry_adapter
from src.geometry_adapter import InitFigureError

IMG_HEIGHT, IMG_WIDTH = 20, 20


def polygon_label(exterior, interior=(), figure_id=1, class_title="cat", geometry_type="polygon"):
    return {
        "id": figure_id,
        "classTitle": class_title,
        "geometryType": geometry_type,
        "points": {
            "exterior": [list(point) for point in exterior],
            "interior": [[list(point) for point in contour] for contour in interior],
        },
    }


def multipolygon_label(parts, figure_id=1, class_title="cat", geometry_type="multipolygon"):
    return {
        "id": figure_id,
        "classTitle": class_title,
        "geometryType": geometry_type,
        "parts": [
            {
                "exterior": [list(point) for point in exterior],
                "interior": [[list(point) for point in contour] for contour in interior],
            }
            for exterior, interior in parts
        ],
    }


def bitmap_label(mask, origin_row, origin_col, figure_id=1, class_title="cat"):
    bitmap = sly.Bitmap(data=mask, origin=sly.PointLocation(row=origin_row, col=origin_col))
    label = {"id": figure_id, "classTitle": class_title, "geometryType": "bitmap"}
    label.update(bitmap.to_json())
    return label


def square(left, top, right, bottom):
    """Closed ring of an axis aligned square in [x, y] image coordinates."""
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def rendered(bitmap, height=IMG_HEIGHT, width=IMG_WIDTH):
    """Full-image mask exactly as it is handed to the predictor."""
    return functional.bitmap_to_mask(bitmap, height, width) > 0


class FakeAnnotationApi:
    def __init__(self, ann_json):
        self.ann_json = ann_json
        self.downloaded = []

    def download_json(self, image_id):
        self.downloaded.append(image_id)
        return self.ann_json


class FakeImageInfo:
    def __init__(self, height, width):
        self.height = height
        self.width = width


class FakeImageApi:
    def __init__(self, height, width):
        self.info = FakeImageInfo(height, width)
        self.requested = []

    def get_info_by_id(self, image_id):
        self.requested.append(image_id)
        return self.info


class FakeApi:
    def __init__(self, ann_json, height=IMG_HEIGHT, width=IMG_WIDTH):
        self.annotation = FakeAnnotationApi(ann_json)
        self.image = FakeImageApi(height, width)


def annotation(labels, height=IMG_HEIGHT, width=IMG_WIDTH, with_size=True):
    ann_json = {"description": "", "tags": [], "objects": list(labels)}
    if with_size:
        ann_json["size"] = {"height": height, "width": width}
    return ann_json


# --- polygons -----------------------------------------------------------------


def test_polygon_with_hole_is_rasterized_with_correct_origin():
    label = polygon_label(square(2, 2, 11, 11), interior=[square(5, 5, 8, 8)])

    bitmap = geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)

    assert (bitmap.origin.row, bitmap.origin.col) == (2, 2)
    assert bitmap.data.shape == (10, 10)
    mask = rendered(bitmap)
    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[2:12, 2:12] = True
    expected[5:9, 5:9] = False  # the hole
    assert np.array_equal(mask, expected)


def test_polygon_of_an_any_shape_class_is_supported():
    label = polygon_label(square(3, 4, 6, 9), class_title="any_shape_object")

    mask = rendered(geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH))

    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[4:10, 3:7] = True
    assert np.array_equal(mask, expected)


def test_polygon_is_clipped_at_the_image_bounds():
    label = polygon_label(square(-5, -5, 4, 4))

    bitmap = geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)

    assert (bitmap.origin.row, bitmap.origin.col) == (0, 0)
    mask = rendered(bitmap)
    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[0:5, 0:5] = True
    assert np.array_equal(mask, expected)


def test_polygon_touching_the_last_row_and_column_is_kept():
    label = polygon_label(square(15, 15, IMG_WIDTH + 5, IMG_HEIGHT + 5))

    mask = rendered(geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH))

    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[15:IMG_HEIGHT, 15:IMG_WIDTH] = True
    assert np.array_equal(mask, expected)


def test_polygon_outside_of_the_image_is_reported():
    label = polygon_label(square(40, 40, 50, 50))

    with pytest.raises(InitFigureError):
        geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)


# --- multipolygons ------------------------------------------------------------


def test_multipolygon_keeps_disconnected_parts():
    label = multipolygon_label([(square(1, 1, 4, 4), []), (square(10, 12, 14, 16), [])])

    bitmap = geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)

    assert (bitmap.origin.row, bitmap.origin.col) == (1, 1)
    mask = rendered(bitmap)
    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[1:5, 1:5] = True
    expected[12:17, 10:15] = True
    assert np.array_equal(mask, expected)


@pytest.mark.parametrize("holed_part_first", [True, False])
def test_hole_of_one_part_does_not_erase_an_overlapping_part(holed_part_first):
    holed = (square(2, 2, 11, 11), [square(5, 5, 8, 8)])
    solid = (square(6, 6, 15, 15), [])
    parts = [holed, solid] if holed_part_first else [solid, holed]
    label = multipolygon_label(parts)

    mask = rendered(geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH))

    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[2:12, 2:12] = True
    expected[5:9, 5:9] = False
    expected[6:16, 6:16] = True  # the second part refills the overlapped hole pixels
    assert np.array_equal(mask, expected)
    assert mask[7, 7]  # inside the hole, covered by the second part
    assert not mask[5, 5]  # inside the hole, not covered by the second part


def test_multipolygon_of_an_any_shape_class_is_clipped_at_the_image_bounds():
    label = multipolygon_label(
        [(square(-3, -3, 2, 2), []), (square(17, 17, 30, 30), [])],
        class_title="any_shape_object",
    )

    mask = rendered(geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH))

    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[0:3, 0:3] = True
    expected[17:IMG_HEIGHT, 17:IMG_WIDTH] = True
    assert np.array_equal(mask, expected)


# --- bitmaps (existing behavior) ----------------------------------------------


def test_bitmap_label_keeps_its_origin():
    data = np.zeros((4, 6), bool)
    data[1:3, 2:5] = True
    label = bitmap_label(data, origin_row=7, origin_col=3)

    bitmap = geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)

    mask = rendered(bitmap)
    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[8:10, 5:8] = True
    assert np.array_equal(mask, expected)


def test_legacy_bitmap_label_without_geometry_type_is_still_read():
    data = np.ones((3, 3), bool)
    label = bitmap_label(data, origin_row=2, origin_col=2)
    del label["geometryType"]

    mask = rendered(geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH))

    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[2:5, 2:5] = True
    assert np.array_equal(mask, expected)


def test_bitmap_hanging_over_the_image_border_is_clipped():
    data = np.ones((6, 6), bool)
    label = bitmap_label(data, origin_row=IMG_HEIGHT - 2, origin_col=IMG_WIDTH - 3)

    bitmap = geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)

    assert bitmap.data.shape == (2, 3)
    mask = rendered(bitmap)  # would raise without clipping
    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    expected[IMG_HEIGHT - 2 :, IMG_WIDTH - 3 :] = True
    assert np.array_equal(mask, expected)


def test_bitmap_fully_outside_of_the_image_is_reported():
    label = bitmap_label(np.ones((3, 3), bool), origin_row=IMG_HEIGHT + 1, origin_col=0)

    with pytest.raises(InitFigureError):
        geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)


# --- unsupported and malformed geometry ---------------------------------------


def test_unsupported_geometry_is_not_parsed_as_a_bitmap():
    label = {
        "id": 1,
        "classTitle": "cat",
        "geometryType": "rectangle",
        "points": {"exterior": [[1, 1], [5, 5]], "interior": []},
    }

    with pytest.raises(InitFigureError) as error:
        geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)
    assert "rectangle" in str(error.value)


def test_geometry_type_of_an_ambiguous_label_is_not_guessed():
    label = {"id": 1, "classTitle": "cat"}

    with pytest.raises(InitFigureError):
        geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)


@pytest.mark.parametrize(
    "label",
    [
        pytest.param(polygon_label([(1, 1), (5, 5)]), id="too_few_points"),
        pytest.param(polygon_label([(1, 1), (5, 5), ("x", 7)]), id="not_a_number"),
        pytest.param(polygon_label([(1, 1), (5, 5), (7, 7, 7)]), id="not_a_pair"),
        pytest.param({"id": 1, "geometryType": "polygon"}, id="no_points"),
        pytest.param({"id": 1, "geometryType": "multipolygon", "parts": []}, id="no_parts"),
        pytest.param(
            {"id": 1, "geometryType": "multipolygon", "parts": [[[1, 1], [5, 5], [7, 7]]]},
            id="part_is_not_an_object",
        ),
        pytest.param(
            {"id": 1, "geometryType": "multipolygon", "parts": [{"interior": []}]},
            id="part_without_exterior",
        ),
    ],
)
def test_malformed_geometry_is_reported(label):
    with pytest.raises(InitFigureError):
        geometry_adapter.label_to_bitmap(label, IMG_HEIGHT, IMG_WIDTH)


# --- download_init_mask -------------------------------------------------------


def test_download_init_mask_uses_the_annotation_size():
    label = polygon_label(square(2, 2, 5, 5), figure_id=777)
    api = FakeApi(annotation([{"id": 1, "geometryType": "bitmap"}, label]))

    bitmap = geometry_adapter.download_init_mask(api, figure_id=777, image_id=42)

    assert api.annotation.downloaded == [42]
    assert api.image.requested == []  # the annotation already carries the image size
    assert (bitmap.origin.row, bitmap.origin.col) == (2, 2)
    assert bitmap.data.shape == (4, 4)


def test_download_init_mask_falls_back_to_the_image_info():
    label = polygon_label(square(2, 2, 30, 30), figure_id=777)
    api = FakeApi(annotation([label], with_size=False), height=8, width=9)

    bitmap = geometry_adapter.download_init_mask(api, figure_id=777, image_id=42)

    assert api.image.requested == [42]
    assert bitmap.data.shape == (6, 7)  # clipped to the 8x9 image


def test_download_init_mask_reports_a_missing_figure():
    api = FakeApi(annotation([polygon_label(square(2, 2, 5, 5), figure_id=1)]))

    with pytest.raises(InitFigureError) as error:
        geometry_adapter.download_init_mask(api, figure_id=2, image_id=42)
    assert "2" in str(error.value)


@pytest.mark.parametrize("figure_id,image_id", [(None, 42), (777, None)])
def test_download_init_mask_reports_missing_ids(figure_id, image_id):
    api = FakeApi(annotation([polygon_label(square(2, 2, 5, 5), figure_id=777)]))

    with pytest.raises(InitFigureError):
        geometry_adapter.download_init_mask(api, figure_id=figure_id, image_id=image_id)
