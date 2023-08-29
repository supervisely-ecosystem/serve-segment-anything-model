import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import threading
import time
from cacheout import Cache
from dotenv import load_dotenv

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from fastapi import Response, Request, status
from pathlib import Path


import supervisely as sly
from supervisely.imaging.color import generate_rgb
from supervisely.app.widgets import RadioGroup, Field
from supervisely.nn.inference.interactive_segmentation import functional
from supervisely.sly_logger import logger
from supervisely.imaging import image as sly_image
from supervisely.io.fs import silent_remove
from supervisely._utils import rand_str
from supervisely.app.content import get_data_dir


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])
weights_location_path = "/weights"
model_data_path = os.path.join(root_source_path, "models", "model_data.json")
api = sly.Api()


class SegmentAnythingModel(sly.nn.inference.PromptableSegmentation):
    def add_content_to_custom_tab(self, gui):
        self.select_task_type = RadioGroup(
            items=[
                RadioGroup.Item(value="vit_b"),
                RadioGroup.Item(value="vit_l"),
                RadioGroup.Item(value="vit_h"),
            ],
            direction="vertical",
        )
        select_task_type_f = Field(self.select_task_type, "Select model architecture")
        return select_task_type_f

    def get_models(self, mode="table"):
        model_data = sly.json.load_json_file(model_data_path)
        if mode == "table":
            for element in model_data:
                del element["weights_link"]
            return model_data
        elif mode == "links":
            models_data_processed = {}
            for element in model_data:
                models_data_processed[element["Model"]] = {"weights_link": element["weights_link"]}
            return models_data_processed

    def download_weights(self, model_dir):
        model_source = self.gui.get_model_source()
        if model_source == "Pretrained models":
            models_data = self.get_models(mode="links")
            selected_model = self.gui.get_checkpoint_info()["Model"]
            weights_link = models_data[selected_model]["weights_link"]
            weights_file_name = selected_model.replace(" ", "_") + ".pth"
            weights_dst_path = os.path.join(weights_location_path, weights_file_name)
            # for debug
            # weights_dst_path = os.path.join(model_dir, weights_file_name)
            if not sly.fs.file_exists(weights_dst_path):
                self.download(src_path=weights_link, dst_path=weights_dst_path)
        elif model_source == "Custom models":
            custom_link = self.gui.get_custom_link()
            weights_file_name = os.path.basename(custom_link)
            weights_dst_path = os.path.join(model_dir, weights_file_name)
            if not sly.fs.file_exists(weights_dst_path):
                self.download(
                    src_path=custom_link,
                    dst_path=weights_dst_path,
                )
        return weights_dst_path

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # get weights path
        weights_path = self.download_weights(model_dir)
        # get model name
        model_source = self.gui.get_model_source()
        if model_source == "Pretrained models":
            selected_model = self.gui.get_checkpoint_info()["Model"]
            model_name = selected_model.lower().replace("-", "_")[:5]
        elif model_source == "Custom models":
            model_name = self.select_task_type.get_value()
        # build model
        self.sam = sam_model_registry[model_name](checkpoint=weights_path)
        # load model on device
        if device != "cpu":
            if device == "cuda":
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(int(device[-1]))
            torch_device = torch.device(device)
            self.sam.to(device=torch_device)
        else:
            self.sam.to(device=device)
        # build predictor
        self.predictor = SamPredictor(self.sam)
        # define class names
        self.class_names = ["target_mask"]
        # list for storing mask colors
        self.mask_colors = [[255, 0, 0]]
        # variable for storing image ids from previous inference iterations
        self.previous_image_id = None
        # dict for storing model variables to avoid unnecessary calculations
        self.cache = Cache(maxsize=100, ttl=5 * 60)
        # set variables for smart tool mode
        self._inference_image_lock = threading.Lock()

        # TODO: add maxsize after discuss
        self._inference_image_cache = Cache(ttl=60)

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        return info

    def get_classes(self) -> List[str]:
        return self.class_names

    @property
    def model_meta(self):
        if self._model_meta is None:
            self._model_meta = sly.ProjectMeta(
                [sly.ObjClass(self.class_names[0], sly.Bitmap, [255, 0, 0])]
            )
            self._get_confidence_tag_meta()
        return self._model_meta

    def set_image_data(self, input_image, settings):
        if settings["input_image_id"] != self.previous_image_id:
            if settings["input_image_id"] not in self.cache:
                self.predictor.set_image(input_image)
                self.cache.set(
                    settings["input_image_id"],
                    {
                        "features": self.predictor.features,
                        "input_size": self.predictor.input_size,
                        "original_size": self.predictor.original_size,
                    },
                )
            else:
                cached_data = self.cache.get(settings["input_image_id"])
                self.predictor.features = cached_data["features"]
                self.predictor.input_size = cached_data["input_size"]
                self.predictor.original_size = cached_data["original_size"]

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionMask]:
        # prepare input data
        input_image = sly.image.read(image_path)
        # list for storing preprocessed masks
        predictions = []
        if self._model_meta is None:
            self._model_meta = self.model_meta
        if settings["mode"] == "raw":
            # build mask generator and generate masks
            mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=settings["points_per_side"],
                points_per_batch=settings["points_per_batch"],
                pred_iou_thresh=settings["pred_iou_thresh"],
                stability_score_thresh=settings["stability_score_thresh"],
                stability_score_offset=settings["stability_score_offset"],
                box_nms_thresh=settings["box_nms_thresh"],
                crop_n_layers=settings["crop_n_layers"],
                crop_nms_thresh=settings["crop_nms_thresh"],
                crop_overlap_ratio=settings["crop_overlap_ratio"],
                crop_n_points_downscale_factor=settings["crop_n_points_downscale_factor"],
                min_mask_region_area=settings["min_mask_region_area"],
                output_mode=settings["output_mode"],
            )
            masks = mask_generator.generate(input_image)
            for i, mask in enumerate(masks):
                class_name = "object_" + str(i)
                # add new class to model meta if necessary
                if not self._model_meta.get_obj_class(class_name):
                    color = generate_rgb(self.mask_colors)
                    self.mask_colors.append(color)
                    self.class_names.append(class_name)
                    new_class = sly.ObjClass(class_name, sly.Bitmap, color)
                    self._model_meta = self._model_meta.add_obj_class(new_class)
                # get predicted mask
                mask = mask["segmentation"]
                predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        elif settings["mode"] == "bbox":
            # get bbox coordinates
            if "rectangle" not in settings:
                bbox_coordinates = settings["bbox_coordinates"]
            else:
                rectangle = sly.Rectangle.from_json(settings["rectangle"])
                bbox_coordinates = [
                    rectangle.top,
                    rectangle.left,
                    rectangle.bottom,
                    rectangle.right,
                ]
            # transform bbox from yxyx to xyxy format
            bbox_coordinates = [
                bbox_coordinates[1],
                bbox_coordinates[0],
                bbox_coordinates[3],
                bbox_coordinates[2],
            ]
            bbox_coordinates = np.array(bbox_coordinates)
            # get bbox class name and add new class to model meta if necessary
            class_name = settings["bbox_class_name"] + "_mask"
            if not self._model_meta.get_obj_class(class_name):
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Bitmap, [255, 0, 0])
                self._model_meta = self._model_meta.add_obj_class(new_class)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            self.set_image_data(input_image, settings)
            self.previous_image_id = settings["input_image_id"]
            # get predicted mask
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_coordinates[None, :],
                multimask_output=False,
            )
            mask = masks[0]
            predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        elif settings["mode"] == "points":
            # get point coordinates
            point_coordinates = settings["point_coordinates"]
            point_coordinates = np.array(point_coordinates)
            # get point labels
            point_labels = settings["point_labels"]
            point_labels = np.array(point_labels)
            # set class name
            if settings["points_class_name"]:
                class_name = settings["points_class_name"]
            else:
                class_name = self.class_names[0]
            # add new class to model meta if necessary
            if not self._model_meta.get_obj_class(class_name):
                color = generate_rgb(self.mask_colors)
                self.mask_colors.append(color)
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Bitmap, color)
                self._model_meta = self._model_meta.add_obj_class(new_class)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            self.set_image_data(input_image, settings)
            self.previous_image_id = settings["input_image_id"]
            # get predicted masks
            masks, _, _ = self.predictor.predict(
                point_coords=point_coordinates,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = masks[0]
            predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        elif settings["mode"] == "combined":
            # get point coordinates
            point_coordinates = settings["point_coordinates"]
            point_coordinates = np.array(point_coordinates)
            # get point labels
            point_labels = settings["point_labels"]
            point_labels = np.array(point_labels)
            # get bbox coordinates
            bbox_coordinates = settings["bbox_coordinates"]
            # transform bbox from yxyx to xyxy format
            bbox_coordinates = [
                bbox_coordinates[1],
                bbox_coordinates[0],
                bbox_coordinates[3],
                bbox_coordinates[2],
            ]
            bbox_coordinates = np.array(bbox_coordinates)
            # get bbox class name and add new class to model meta if necessary
            class_name = settings["bbox_class_name"] + "_mask"
            if not self._model_meta.get_obj_class(class_name):
                self.class_names.append(class_name)
                new_class = sly.ObjClass(class_name, sly.Bitmap, [255, 0, 0])
                self._model_meta = self._model_meta.add_obj_class(new_class)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            self.set_image_data(input_image, settings)
            # get predicted masks
            if (
                settings["input_image_id"] in self.cache
                and (
                    self.cache.get(settings["input_image_id"]).get("previous_bbox")
                    == bbox_coordinates
                ).all()
                and self.previous_image_id == settings["input_image_id"]
            ):
                # get mask from previous predicton and use at as an input for new prediction
                mask_input = self.cache.get(settings["input_image_id"])["mask_input"]
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coordinates,
                    point_labels=point_labels,
                    box=bbox_coordinates[None, :],
                    mask_input=mask_input[None, :, :],
                    multimask_output=False,
                )
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coordinates,
                    point_labels=point_labels,
                    box=bbox_coordinates[None, :],
                    multimask_output=False,
                )
            # save bbox ccordinates and mask to cache
            if settings["input_image_id"] in self.cache:
                image_id = settings["input_image_id"]
                cached_data = self.cache.get(image_id)
                cached_data["previous_bbox"] = bbox_coordinates
                cached_data["mask_input"] = logits[0]
                self.cache.set(image_id, cached_data)
            # update previous_image_id variable
            self.previous_image_id = settings["input_image_id"]
            mask = masks[0]
            predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        return predictions

    def serve(self):
        super().serve()
        server = self._app.get_server()
        self.add_cache_endpoint(server)

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
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
                settings = self._get_inference_settings(state)
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

            if hash_str not in self._inference_image_cache:
                logger.debug(f"downloading image: {hash_str}")
                image_np = functional.download_image_from_context(
                    smtool_state,
                    api,
                    app_dir,
                    cache_load_img=self.download_image,
                    cache_load_frame=self.download_frame,
                    cache_load_img_hash=self.download_image_by_hash,
                )
                self._inference_image_cache.set(hash_str, image_np)
            else:
                logger.debug(f"image found in cache: {hash_str}")
                image_np = self._inference_image_cache.get(hash_str)

            # crop
            image_path = os.path.join(app_dir, f"{time.time()}_{rand_str(10)}.jpg")
            if isinstance(image_np, list):
                image_np = image_np[0]
            sly_image.write(image_path, image_np)

            self._inference_image_lock.acquire()
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
                    crop[1]["y"],
                    crop[1]["x"],
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
                pred_mask = self.predict(image_path, settings)[0].mask
            finally:
                logger.debug("Predict done")
                self._inference_image_lock.release()
                silent_remove(image_path)

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

        @server.post("/is_online")
        def is_online(response: Response, request: Request):
            response = {"is_online": True}
            return response
        
        @server.post("/smart_segmentation_batched")
        def smart_segmentation_batched(response: Response, request: Request):
            response_batch = {}
            data = request.state.context["data_to_process"]
            app_session_id = sly.io.env.task_id()
            for image_idx, image_data in data.items():
                image_prediction = api.task.send_request(
                    app_session_id,
                    "smart_segmentation",
                    data={},
                    context=image_data,
                )
                response_batch[image_idx] = image_prediction
            return response_batch


m = SegmentAnythingModel(
    use_gui=True,
    custom_inference_settings=os.path.join(root_source_path, "custom_settings.yaml"),
)

if sly.is_production():
    m.serve()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    settings = {}
    settings["mode"] = "bbox"
    settings["input_image_id"] = 19491102
    settings["bbox_coordinates"] = [706, 393, 967, 1112]
    settings["bbox_class_name"] = "raven"
    results = m.predict(image_path, settings=settings)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=7)
    print(f"predictions and visualization have been saved: {vis_path}")
