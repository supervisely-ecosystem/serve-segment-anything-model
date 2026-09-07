import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import threading
from cacheout import Cache
from dotenv import load_dotenv

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from fastapi import Response, Request
from pathlib import Path
from cachetools import LRUCache


import supervisely as sly
from supervisely.imaging.color import generate_rgb
from supervisely.app.widgets import RadioGroup, Field

from src import smart_tool


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
        self.class_names = ["object_mask"]
        # list for storing mask colors
        self.mask_colors = [[255, 0, 0]]
        # variable for storing image ids from previous inference iterations
        self.previous_image_id = None
        # dict for storing model variables to avoid unnecessary calculations
        self.model_cache = Cache(maxsize=100, ttl=5 * 60)
        # set variables for smart tool mode
        self._inference_image_lock = threading.Lock()

        # TODO: add maxsize after discuss
        self._inference_image_cache = Cache(ttl=60)
        self._init_mask_cache = LRUCache(maxsize=100)  # cache of sly.Bitmaps

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = True
        info["async_video_inference_support"] = True
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
            if settings["input_image_id"] not in self.model_cache:
                self.predictor.set_image(input_image)
                self.model_cache.set(
                    settings["input_image_id"],
                    {
                        "features": self.predictor.features,
                        "input_size": self.predictor.input_size,
                        "original_size": self.predictor.original_size,
                    },
                )
            else:
                cached_data = self.model_cache.get(settings["input_image_id"])
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
            for mask in masks:
                # get predicted mask
                mask = mask["segmentation"]
                predictions.append(sly.nn.PredictionMask(class_name="object_mask", mask=mask))
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
            init_mask = settings["init_mask"]
            # get predicted masks
            if (
                settings["input_image_id"] in self.model_cache
                and (
                    self.model_cache.get(settings["input_image_id"]).get("previous_bbox")
                    == bbox_coordinates
                ).all()
                and self.previous_image_id == settings["input_image_id"]
            ):
                # get mask from previous predicton and use at as an input for new prediction
                mask_input = self.model_cache.get(settings["input_image_id"])["mask_input"]
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coordinates,
                    point_labels=point_labels,
                    box=bbox_coordinates[None, :],
                    mask_input=mask_input[None, :, :],
                    multimask_output=False,
                )
            elif init_mask is not None:
                # transform
                mask_input = self.predictor.transform.apply_image(init_mask)
                # pad
                h, w = mask_input.shape[:2]
                padh = self.predictor.model.image_encoder.img_size - h
                padw = self.predictor.model.image_encoder.img_size - w
                mask_input = np.pad(mask_input, ((0, padh), (0, padw)))
                # downscale to 256x256
                mask_input = cv2.resize(mask_input, (256, 256), interpolation=cv2.INTER_LINEAR)
                # put values
                mask_input = mask_input.astype(float)
                mask_input[mask_input > 0] = 20
                mask_input[mask_input <= 0] = -20
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
            if settings["input_image_id"] in self.model_cache:
                image_id = settings["input_image_id"]
                cached_data = self.model_cache.get(image_id)
                cached_data["previous_bbox"] = bbox_coordinates
                cached_data["mask_input"] = logits[0]
                self.model_cache.set(image_id, cached_data)
            # update previous_image_id variable
            self.previous_image_id = settings["input_image_id"]
            mask = masks[0]
            predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        return predictions

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/smart_segmentation")
        def smart_segmentation(response: Response, request: Request):
            return smart_tool.smart_segmentation(self, response, request)

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
