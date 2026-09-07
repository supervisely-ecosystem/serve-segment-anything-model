"""Implementation of the Smart Tool `/smart_segmentation` route.

Kept out of `src/main.py` so that request parsing, initial figure normalization,
the handoff to the predictor and the response encoding can be exercised offline,
without importing the model stack, loading weights or starting the service.
"""

import os
import time

from fastapi import status

import supervisely as sly
from supervisely._utils import rand_str
from supervisely.app.content import get_data_dir
from supervisely.imaging import image as sly_image
from supervisely.io.fs import silent_remove
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.sly_logger import logger

from src import geometry_adapter


def smart_segmentation(model, response, request):
    """Predict a mask from clicks, optionally initialized with an existing figure.

    `model` is the served `SegmentAnythingModel`: only its caches, its inference
    lock and its `predict` method are used here.
    """
    # 1. parse request
    # 2. download image
    # 3. make crop
    # 4. predict

    logger.debug(
        f"smart_segmentation inference: context=",
        extra={**request.state.context},
    )

    try:
        state = request.state.state
        settings = model._get_inference_settings(state)
        smtool_state = request.state.context
        api = request.state.api
        crop = smtool_state["crop"]
        positive_clicks, negative_clicks = (
            smtool_state["positive"],
            smtool_state["negative"],
        )
        if len(positive_clicks) + len(negative_clicks) == 0:
            logger.warn("No clicks received.")
            response = {
                "origin": None,
                "bitmap": None,
                "success": True,
                "error": None,
            }
            return response
    except Exception as exc:
        logger.warn("Error parsing request:" + str(exc), exc_info=True)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"message": "400: Bad request.", "success": False}

    # collect clicks
    uncropped_clicks = [{**click, "is_positive": True} for click in positive_clicks]
    uncropped_clicks += [{**click, "is_positive": False} for click in negative_clicks]
    clicks = functional.transform_clicks_to_crop(crop, uncropped_clicks)
    is_in_bbox = functional.validate_click_bounds(crop, clicks)
    if not is_in_bbox:
        logger.warn(f"Invalid value: click is out of bbox bounds.")
        return {
            "origin": None,
            "bitmap": None,
            "success": True,
            "error": None,
        }

    # download image if needed (using cache)
    app_dir = get_data_dir()
    hash_str = functional.get_hash_from_context(smtool_state)

    if hash_str not in model._inference_image_cache:
        logger.debug(f"Downloading image: {hash_str} to local cache")
        t = time.monotonic()
        image_np = functional.download_image_from_context(
            smtool_state,
            api,
            app_dir,
            cache_load_img=model.cache.download_image,
            cache_load_frame=model.cache.download_frame,
            cache_load_img_hash=model.cache.download_image_by_hash,
        )
        logger.debug(f"Image {hash_str} downloaded in {time.monotonic() - t:.3f} sec")
        t = time.monotonic()
        model._inference_image_cache.set(hash_str, image_np)
        logger.debug(f"Image #{hash_str} added to local cache in {time.monotonic() - t:.3f} sec")
    else:
        logger.debug(f"image found in cache: {hash_str}")
        image_np = model._inference_image_cache.get(hash_str)

    # crop
    t = time.monotonic()
    image_path = os.path.join(app_dir, f"{time.time()}_{rand_str(10)}.jpg")
    if isinstance(image_np, list):
        image_np = image_np[0]
    sly_image.write(image_path, image_np)
    logger.debug(f"image saved to disk: {image_path} in {time.monotonic() - t:.3f} sec")

    t = time.monotonic()
    # Prepare init_mask (only for images)
    figure_id = smtool_state.get("figure_id")
    image_id = smtool_state.get("image_id")
    try:
        if smtool_state.get("init_figure") is True and image_id is not None:
            # Download, normalize the concrete geometry and save in Cache
            init_mask = geometry_adapter.download_init_mask(api, figure_id, image_id)
            model._init_mask_cache[figure_id] = init_mask
        elif model._init_mask_cache.get(figure_id) is not None:
            # Load from Cache
            init_mask = model._init_mask_cache[figure_id]
        else:
            init_mask = None
    except geometry_adapter.InitFigureError as exc:
        # Unsupported or malformed initial geometry is reported instead of being
        # parsed as a bitmap or silently dropped.
        logger.warn(f"Can not use the initial figure: {exc}")
        silent_remove(image_path)
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {
            "origin": None,
            "bitmap": None,
            "success": False,
            "error": str(exc),
        }
    if init_mask is not None:
        image_info = api.image.get_info_by_id(image_id)
        init_mask = functional.bitmap_to_mask(init_mask, image_info.height, image_info.width)
        # init_mask = functional.crop_image(crop, init_mask)
        assert init_mask.shape[:2] == image_np.shape[:2]
    settings["init_mask"] = init_mask
    logger.debug(f"init_mask prepared in {time.monotonic() - t:.3f} sec")

    t = time.monotonic()
    model._inference_image_lock.acquire()
    try:
        # predict
        logger.debug("Preparing settings for inference request...")
        settings["mode"] = "combined"
        if "image_id" in smtool_state:
            settings["input_image_id"] = smtool_state["image_id"]
        elif "video" in smtool_state:
            settings["input_image_id"] = hash_str
        elif "image_hash" in smtool_state:
            settings["input_image_id"] = smtool_state["image_hash"]
        settings["bbox_coordinates"] = [
            crop[0]["y"],
            crop[0]["x"],
            crop[1]["y"] + 1,
            crop[1]["x"] + 1,
        ]
        settings["bbox_class_name"] = "target"
        point_coordinates, point_labels = [], []
        for click in uncropped_clicks:
            point_coordinates.append([click["x"], click["y"]])
            if click["is_positive"]:
                point_labels.append(1)
            else:
                point_labels.append(0)
        settings["point_coordinates"], settings["point_labels"] = (
            point_coordinates,
            point_labels,
        )
        pred_mask = model.predict(image_path, settings)[0].mask
    finally:
        logger.debug("Predict done")
        model._inference_image_lock.release()
        silent_remove(image_path)
    logger.debug(f"smart_segmentation inference done in {time.monotonic() - t:.3f} sec")

    if pred_mask.any():
        bitmap = sly.Bitmap(pred_mask)
        # crop bitmap
        bitmap = bitmap.crop(sly.Rectangle(*settings["bbox_coordinates"]))[0]
        # adapt bitmap to crop coordinates
        bitmap_data = bitmap.data
        bitmap_origin = sly.PointLocation(
            bitmap.origin.row - crop[0]["y"],
            bitmap.origin.col - crop[0]["x"],
        )
        bitmap = sly.Bitmap(data=bitmap_data, origin=bitmap_origin)
        bitmap_origin, bitmap_data = functional.format_bitmap(bitmap, crop)
        logger.debug(f"smart_segmentation inference done!")
        response = {
            "origin": bitmap_origin,
            "bitmap": bitmap_data,
            "success": True,
            "error": None,
        }
    else:
        logger.debug(f"Predicted mask is empty.")
        response = {
            "origin": None,
            "bitmap": None,
            "success": True,
            "error": None,
        }
    return response
