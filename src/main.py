import supervisely as sly
from supervisely.imaging.color import generate_rgb
from supervisely.app.widgets import RadioGroup, Field
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

try:
    from typing import Literal
except ImportError:
    # for compatibility with python 3.7
    from typing_extensions import Literal
from typing import List, Any, Dict
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[1])
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
            weights_dst_path = os.path.join(model_dir, weights_file_name)
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
        self.sam.to(device=device)
        # define class names
        self.class_names = ["target"]
        # list for storing mask colors
        self.mask_colors = [[255, 0, 0]]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

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

    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionMask]:
        # prepare input data
        input_image = sly.image.read(image_path)
        # list for storing preprocessed masks
        predictions = []
        # list for storing image ids
        image_ids = []
        if not sly.is_production():
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
            # build predictor
            predictor = SamPredictor(self.sam)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            if settings["input_image_id"] not in image_ids:
                predictor.set_image(input_image)
                image_ids.append(settings["input_image_id"])
            # get predicted mask
            masks, _, _ = predictor.predict(
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
            class_name = self.class_names[0]
            # build predictor
            predictor = SamPredictor(self.sam)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            if settings["input_image_id"] not in image_ids:
                predictor.set_image(input_image)
                image_ids.append(settings["input_image_id"])
            # get predicted masks
            masks, _, _ = predictor.predict(
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
            # build predictor
            predictor = SamPredictor(self.sam)
            # generate image embedding - model will remember this embedding and use it for subsequent mask prediction
            if settings["input_image_id"] not in image_ids:
                predictor.set_image(input_image)
                image_ids.append(settings["input_image_id"])
            # get predicted masks
            masks, _, _ = predictor.predict(
                point_coords=point_coordinates,
                point_labels=point_labels,
                box=bbox_coordinates[None, :],
                multimask_output=False,
            )
            mask = masks[0]
            predictions.append(sly.nn.PredictionMask(class_name=class_name, mask=mask))
        return predictions


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
