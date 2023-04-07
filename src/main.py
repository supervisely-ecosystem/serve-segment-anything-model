import supervisely as sly
from supervisely.imaging.color import random_rgb, generate_rgb
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

class SegmentAnythingModel(sly.nn.inference.SemanticSegmentation):
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
            model_name = selected_model.lower().replace("-", '_')[:5]
        elif model_source == "Custom models":
            model_name = self.select_task_type.get_value()
        # build model
        self.sam = sam_model_registry[model_name](checkpoint=weights_path)
        # load model on device
        self.sam.to(device=device)
        # define class names
        self.class_names = ["object_mask"]
        # list for storing mask colors
        self.mask_colors = []
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_info(self):
        info = super().get_info()
        info["task type"] = "promptable segmentation"
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
        
    def predict(self, image_path: str, settings: Dict[str, Any]) -> List[sly.nn.PredictionBBox]:
        # prepare input data
        input_image = sly.image.read(image_path)
        # list for storing preprocessed masks
        if settings["mode"] == "raw":
            mask_generator = SamAutomaticMaskGenerator(self.sam)
            unprocessed_masks = mask_generator.generate(input_image)
            for mask in unprocessed_masks:
                mask = mask["segmentation"]
        return []
    

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
    settings["mode"] = "raw"
    results = m.predict(image_path, settings=settings)
    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=7)
    print(f"predictions and visualization have been saved: {vis_path}")