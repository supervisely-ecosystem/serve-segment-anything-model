"""Offline tests for the `/smart_segmentation` route.

The production request handling is executed for real: only the instance API and
the model are replaced at their boundaries, so no weights are loaded, no GPU is
initialized and the service is not started.
"""

import os
import threading
from copy import deepcopy

import numpy as np
import pytest

import supervisely as sly

from src import smart_tool
from tests.unit.test_geometry_adapter import (
    FakeApi,
    annotation,
    bitmap_label,
    multipolygon_label,
    polygon_label,
    square,
)

IMG_HEIGHT, IMG_WIDTH = 20, 20
FIGURE_ID = 777
IMAGE_ID = 42
CROP = [{"x": 2, "y": 3}, {"x": 17, "y": 16}]


class ImageCache:
    """Minimal stand-in for the `cacheout.Cache` of downloaded images."""

    def __init__(self):
        self.storage = {}

    def __contains__(self, key):
        return key in self.storage

    def set(self, key, value):
        self.storage[key] = value

    def get(self, key):
        return self.storage.get(key)


class DownloadCache:
    def __init__(self, image_np):
        self.image_np = image_np
        self.downloads = []

    def download_image(self, api, image_id, related=False):
        self.downloads.append(image_id)
        return self.image_np

    def download_frame(self, api, video_id, frame_index):
        raise AssertionError("Video frames are not expected in these tests")

    def download_image_by_hash(self, api, image_hash):
        raise AssertionError("Image hashes are not expected in these tests")


class FakeModel:
    """The parts of `SegmentAnythingModel` the route interacts with."""

    def __init__(self, image_np, pred_mask):
        self.cache = DownloadCache(image_np)
        self._inference_image_cache = ImageCache()
        self._init_mask_cache = {}
        self._inference_image_lock = threading.Lock()
        self.pred_mask = pred_mask
        self.predict_calls = []

    def _get_inference_settings(self, state):
        return {"points_per_side": 32}

    def predict(self, image_path, settings):
        assert os.path.isfile(image_path), "the route must save the image before predicting"
        self.predict_calls.append(deepcopy(settings))
        return [sly.nn.PredictionMask(class_name=settings["bbox_class_name"], mask=self.pred_mask)]


class FakeRequestState:
    def __init__(self, context, api):
        self.context = context
        self.state = {}
        self.api = api


class FakeRequest:
    def __init__(self, context, api):
        self.state = FakeRequestState(context, api)


class FakeResponse:
    def __init__(self):
        self.status_code = None


def context(**overrides):
    smtool_state = {
        "image_id": IMAGE_ID,
        "crop": deepcopy(CROP),
        "positive": [{"x": 8, "y": 9}],
        "negative": [{"x": 4, "y": 5}],
    }
    smtool_state.update(overrides)
    return smtool_state


def image():
    image_np = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
    image_np[:, :, 0] = 200
    return image_np


def prediction_mask():
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), bool)
    mask[10:14, 6:9] = True
    return mask


@pytest.fixture(autouse=True)
def app_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SLY_APP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TASK_ID", raising=False)
    return tmp_path


def call(model, api, smtool_state):
    response = FakeResponse()
    result = smart_tool.smart_segmentation(model, response, FakeRequest(smtool_state, api))
    return result, response


def test_polygon_init_figure_reaches_the_predictor_as_a_full_image_mask(app_data_dir):
    label = polygon_label(square(4, 5, 9, 12), figure_id=FIGURE_ID)
    api = FakeApi(annotation([label]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    result, response = call(model, api, context(figure_id=FIGURE_ID, init_figure=True))

    assert response.status_code is None
    assert result["success"] is True
    settings = model.predict_calls[0]
    init_mask = settings["init_mask"]
    assert init_mask.dtype == np.uint8
    assert init_mask.shape == (IMG_HEIGHT, IMG_WIDTH)
    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    expected[5:13, 4:10] = 255
    assert np.array_equal(init_mask, expected)
    # the prediction request itself is unchanged
    assert settings["mode"] == "combined"
    assert settings["input_image_id"] == IMAGE_ID
    assert settings["bbox_class_name"] == "target"
    assert settings["bbox_coordinates"] == [3, 2, 17, 18]
    assert settings["point_coordinates"] == [[8, 9], [4, 5]]
    assert settings["point_labels"] == [1, 0]
    assert settings["points_per_side"] == 32  # inference settings are preserved


def test_multipolygon_init_figure_is_united_before_prediction():
    label = multipolygon_label(
        # the hole of the second part must not erase the part drawn before it
        [(square(4, 4, 9, 9), []), (square(2, 2, 7, 7), [square(3, 3, 4, 4)])],
        figure_id=FIGURE_ID,
    )
    api = FakeApi(annotation([label]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    call(model, api, context(figure_id=FIGURE_ID, init_figure=True))

    init_mask = model.predict_calls[0]["init_mask"]
    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    expected[2:8, 2:8] = 255
    expected[3:5, 3:5] = 0  # the hole of the second part
    expected[4:10, 4:10] = 255  # except where the first part covers it
    assert np.array_equal(init_mask, expected)


def test_bitmap_init_figure_still_works():
    data = np.ones((4, 5), bool)
    label = bitmap_label(data, origin_row=6, origin_col=7, figure_id=FIGURE_ID)
    api = FakeApi(annotation([label]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    result, response = call(model, api, context(figure_id=FIGURE_ID, init_figure=True))

    assert result["success"] is True
    init_mask = model.predict_calls[0]["init_mask"]
    expected = np.zeros((IMG_HEIGHT, IMG_WIDTH), np.uint8)
    expected[6:10, 7:12] = 255
    assert np.array_equal(init_mask, expected)


def test_subsequent_clicks_reuse_the_cached_initial_figure():
    label = polygon_label(square(4, 5, 9, 12), figure_id=FIGURE_ID)
    api = FakeApi(annotation([label]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    call(model, api, context(figure_id=FIGURE_ID, init_figure=True))
    call(model, api, context(figure_id=FIGURE_ID, positive=[{"x": 8, "y": 9}, {"x": 7, "y": 7}]))

    assert api.annotation.downloaded == [IMAGE_ID], "the figure must be downloaded only once"
    assert len(model.predict_calls) == 2
    assert np.array_equal(model.predict_calls[1]["init_mask"], model.predict_calls[0]["init_mask"])
    assert model.predict_calls[1]["point_coordinates"] == [[8, 9], [7, 7], [4, 5]]
    assert model.cache.downloads == [IMAGE_ID], "the image must be taken from the local cache"


def test_request_without_an_initial_figure_predicts_from_clicks_only():
    api = FakeApi(annotation([]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    result, response = call(model, api, context())

    assert result["success"] is True
    assert model.predict_calls[0]["init_mask"] is None
    assert api.annotation.downloaded == []


def test_unsupported_initial_geometry_is_reported_and_not_predicted():
    label = {
        "id": FIGURE_ID,
        "classTitle": "cat",
        "geometryType": "rectangle",
        "points": {"exterior": [[1, 1], [5, 5]], "interior": []},
    }
    api = FakeApi(annotation([label]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    result, response = call(model, api, context(figure_id=FIGURE_ID, init_figure=True))

    assert result["success"] is False
    assert result["bitmap"] is None and result["origin"] is None
    assert "rectangle" in result["error"]
    assert response.status_code == 400
    assert model.predict_calls == []


def test_missing_initial_figure_is_reported_and_not_predicted():
    api = FakeApi(annotation([]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    result, response = call(model, api, context(figure_id=FIGURE_ID, init_figure=True))

    assert result["success"] is False
    assert response.status_code == 400
    assert model.predict_calls == []


def test_predicted_mask_is_returned_in_image_coordinates():
    api = FakeApi(annotation([]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    result, _ = call(model, api, context())

    assert result == {
        "origin": {"x": 6, "y": 10},
        "bitmap": result["bitmap"],
        "success": True,
        "error": None,
    }
    decoded = sly.Bitmap.base64_2_data(result["bitmap"])
    assert decoded.shape == (4, 3)
    assert decoded.all()


def test_empty_prediction_returns_an_empty_response():
    api = FakeApi(annotation([]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), np.zeros((IMG_HEIGHT, IMG_WIDTH), bool))

    result, response = call(model, api, context())

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert response.status_code is None


def test_request_without_clicks_is_answered_before_predicting():
    api = FakeApi(annotation([]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    result, _ = call(model, api, context(positive=[], negative=[]))

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert model.predict_calls == []


def test_click_outside_of_the_crop_is_rejected():
    api = FakeApi(annotation([]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    result, _ = call(model, api, context(positive=[{"x": 19, "y": 19}], negative=[]))

    assert result == {"origin": None, "bitmap": None, "success": True, "error": None}
    assert model.predict_calls == []


def test_malformed_request_is_answered_with_bad_request():
    api = FakeApi(annotation([]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())
    broken = context()
    del broken["crop"]

    result, response = call(model, api, broken)

    assert result["success"] is False
    assert response.status_code == 400
    assert model.predict_calls == []


def test_temporary_image_files_are_removed():
    api = FakeApi(annotation([]), height=IMG_HEIGHT, width=IMG_WIDTH)
    model = FakeModel(image(), prediction_mask())

    call(model, api, context())

    app_dir = os.environ["SLY_APP_DATA_DIR"]
    assert [name for name in os.listdir(app_dir) if name.endswith(".jpg")] == []
