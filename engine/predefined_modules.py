import re
import torch
import numpy as np
import io
import json
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig,
)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from unidepth.models import UniDepthV2
import groundingdino.datasets.transforms as T

from VADAR.prompts.vqa_prompt import (
    VQA_PROMPT_CLEVR,
    VQA_PROMPT_GQA,
    VQA_PROMPT_GQA_HOLISTIC,
    VQA_PROMPT,
)
from .engine_utils import *
from groundingdino.util.inference import load_model, predict


class PredefinedModule:
    def __init__(self, name, trace_path=None):
        self.trace_path = trace_path
        self.name = name

    def write_trace(self, html):
        if self.trace_path:
            with open(self.trace_path, "a+") as f:
                f.write(f"{html}\n")


class OracleModule(PredefinedModule):
    def __init__(self, name, trace_path=None):
        super().__init__(name, trace_path)
        self.reference_image = None
        self.scene_json = None
        self.oracle = None

    def set_oracle(self, oracle, reference_image, scene_json):
        self.oracle = oracle
        self.reference_image = reference_image
        self.scene_json = scene_json

    def clear_oracle(self):
        self.reference_image = None
        self.scene_json = None
        self.oracle = None


class LocateModule(OracleModule):
    def __init__(
        self,
        dataset,
        grounding_dino=None,
        molmo_processor=None,
        molmo_model=None,
        trace_path=None,
    ):
        super().__init__("loc", trace_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.molmo_processor = molmo_processor
            self.molmo_model = molmo_model
        else:
            self.grounding_dino = grounding_dino
            self.BOX_THRESHOLD = 0.25
            self.TEXT_TRESHOLD = 0.25

    def _extract_points(self, molmo_output, image_w, image_h):
        all_points = []
        for match in re.finditer(
            r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
            molmo_output,
        ):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                x = int(point[0] * image_w)
                y = int(point[1] * image_h)
                all_points.append([x, y])

        # convert all points to int
        return all_points

    def _parse_bounding_boxes(self, boxes, width, height):
        if len(boxes) == 0:
            return []

        bboxes = []
        for box in boxes:
            cx, cy, w, h = box
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            bboxes.append(
                [
                    int(x1 * width),
                    int(y1 * height),
                    int(x2 * width),
                    int(y2 * height),
                ]
            )
        return bboxes

    def transform_image(self, og_image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        og_image = og_image.convert("RGB")
        img = np.asarray(og_image)
        im_t, _ = transform(og_image, None)
        return img, im_t

    def execute_pts(self, image, object_prompt):
        original_object_prompt = object_prompt
        if self.oracle:
            pts = self.oracle.locate(image, object_prompt, self.scene_json)
        else:
            if object_prompt[-1] != "s":
                object_prompt = object_prompt + "s"
            inputs = self.molmo_processor.process(
                images=[image],
                text="point to the " + object_prompt,
            )
            with torch.no_grad():
                inputs = {
                    k: v.to(self.molmo_model.device).unsqueeze(0)
                    for k, v in inputs.items()
                }
                output = self.molmo_model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=self.molmo_processor.tokenizer,
                )
                generated_tokens = output[0, inputs["input_ids"].size(1) :]
                generated_text = self.molmo_processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                pts = self._extract_points(generated_text, image.size[0], image.size[1])

        if len(pts) == 0:
            self.write_trace(f"<p> No points found<p>")
            return []

        # trace
        if self.oracle:
            self.write_trace(f"<p>Locate [Oracle]: {original_object_prompt}<p>")
        else:
            self.write_trace(f"<p>Locate: {original_object_prompt}<p>")
        dotted_im = dotted_image(image, pts)
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        if len(pts) > 1 and original_object_prompt[-1] != 's':
            original_object_prompt += 's'
        self.write_trace(f"<p>{len(pts)} {original_object_prompt} found<p>")
        self.write_trace(f"<p>Points: {pts}<p>")
        return pts

    def execute_bboxs(self, image, object_prompt):
        original_object_prompt = object_prompt
        width, height = image.size
        prompt = f"{object_prompt.replace(' ', '-')} ."
        _, img_gd = self.transform_image(image)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=img_gd,
                caption=prompt,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_TRESHOLD,
                device="cuda:0",
            )
        bboxes = self._parse_bounding_boxes(boxes, width, height)

        if len(bboxes) == 0:
            self.write_trace(f"<p> No objects found<p>")
            return []

        # trace
        if self.oracle:
            self.write_trace(f"<p>Locate [Oracle]: {original_object_prompt}<p>")
        else:
            self.write_trace(f"<p>Locate: {original_object_prompt}<p>")
        boxed_image = box_image(image, bboxes)
        boxed_html = html_embed_image(boxed_image)
        self.write_trace(boxed_html)
        if len(bboxes) > 1 and original_object_prompt[-1] != 's':
            original_object_prompt += 's'
        self.write_trace(f"<p>{len(bboxes)} {original_object_prompt} found<p>")
        self.write_trace(f"<p>Boxes: {bboxes}<p>")

        return bboxes


class VQAModule(OracleModule):
    def __init__(
        self,
        dataset="omni3d",
        sam2_predictor=None,
        device=None,
        trace_path=None,
        api_key_path="./api.key",
    ):
        super().__init__("vqa", trace_path)
        self.generator = Generator("gpt-4o", api_key_path=api_key_path)
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device

    def _get_prompt(self, question, holistic=False):
        if self.dataset == "clevr":
            return VQA_PROMPT_CLEVR.format(question=question)
        elif self.dataset == "gqa":
            if holistic:
                print("using gqa vqa prompt holistic")
                return VQA_PROMPT_GQA_HOLISTIC.format(question=question)
            else:
                return VQA_PROMPT_GQA.format(question=question)
        else:
            return VQA_PROMPT.format(question=question)

    def _get_bbox(self, mask, margin=20):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add margin
        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(mask.shape[0] - 1, rmax + margin)
        cmax = min(mask.shape[1] - 1, cmax + margin)

        return [cmin, rmin, cmax, rmax]

    def execute_pts(self, image, question, x, y):
        if self.oracle:
            answer = self.oracle.answer_question(x, y, question, self.scene_json)
        else:
            if int(x) == 0 and int(y) == 0:
                answer = self.predict(image, question, holistic=True)
                boxed_image = image
            else:
                with torch.no_grad():
                    sam_inpt_pts = np.array([[int(x), int(y)]])
                    sam_inpt_label = np.array([1])  # foreground label
                    self.sam2_predictor.set_image(np.array(image))

                    masks, scores, logits = self.sam2_predictor.predict(
                        point_coords=sam_inpt_pts,
                        point_labels=sam_inpt_label,
                        multimask_output=True,
                    )

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                if scores[1] > 0.3:
                    box1 = self._get_bbox(masks[0])
                    box2 = self._get_bbox(masks[1])
                    box = [
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3]),
                    ]
                else:
                    box = self._get_bbox(masks[0])
                boxed_image = box_image(image, [box])
                answer = self.predict(boxed_image, question)

        # trace
        im_html = html_embed_image(image, 300)
        if self.oracle:
            self.write_trace(f"<p>Question [Oracle]: {question}</p>")
        else:
            self.write_trace(f"<p>Question: {question}</p>")
        if self.oracle:
            dotted_im = dotted_image(image, [[x, y]])
            dotted_im_html = html_embed_image(dotted_im, 300)
            self.write_trace(dotted_im_html)
        else:
            dotted_im = dotted_image(image, [[x, y]])
            dotted_im_html = html_embed_image(dotted_im, 300)
            self.write_trace(dotted_im_html)
        self.write_trace(f"<p>Answer: {answer}<p>")

        return answer.lower()

    def execute_bboxs(self, image, question, bbox):
        if bbox is None:
            answer = self.predict(image, question, holistic=True)
            boxed_image = image
        else:
            boxed_image = box_image(image, [bbox])
            answer = self.predict(boxed_image, question)

        im_html = html_embed_image(image, 300)
        self.write_trace(im_html)
        boxed_im_html = html_embed_image(boxed_image, 300)
        self.write_trace(boxed_im_html)
        self.write_trace(f"<p>{answer}<p>")
        return answer.lower()

    def remove_substring(self, output, substring):
        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def predict(self, img, question, holistic=False):
        prompt = self._get_prompt(question, holistic)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ]
        output, _ = self.generator.generate("", messages)
        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")
        answer = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].lower()
        return answer


class DepthModule(OracleModule):
    def __init__(
        self,
        unidepth_model,
        device,
        trace_path=None,
    ):
        super().__init__("depth", trace_path)
        self.unidepth_model = unidepth_model
        self.device = device

    def execute_pts(self, image, x, y):
        if self.oracle:
            depth = self.oracle.depth(x, y, self.scene_json)
        else:
            with torch.no_grad():
                rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(self.device)
                preds = self.unidepth_model.infer(rgb)["depth"].squeeze().cpu().numpy()
                depth = preds[int(y), int(x)]
        if self.oracle:
            self.write_trace(f"<p>Get Depth [Oracle]: ({x}, {y})<p>")
        else:
            self.write_trace(f"<p>Get Depth: ({x}, {y})<p>")
        dotted_im = dotted_image(image, [[x, y]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        dotted_im = dotted_image(preds, [[x, y]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        self.write_trace(f"<p>Depth: {depth}<p>")
        return depth

    def execute_bboxs(self, image, bbox):
        x_mid = (bbox[0] + bbox[2]) / 2
        y_mid = (bbox[1] + bbox[3]) / 2
        with torch.no_grad():
            rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(self.device)
            preds = self.unidepth_model.infer(rgb)["depth"].squeeze().cpu().numpy()
            depth = preds[int(y_mid), int(x_mid)]
        if self.oracle:
            self.write_trace(f"<p>Depth [Oracle]: ({x_mid}, {y_mid})<p>")
        else:
            self.write_trace(f"<p>Depth: ({x_mid}, {y_mid})<p>")
        dotted_im = dotted_image(image, [[x_mid, y_mid]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        dotted_im = dotted_image(preds, [[x_mid, y_mid]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        self.write_trace(f"<p>Depth: {depth}<p>")
        return depth


class SameObjectModule(OracleModule):
    def __init__(
        self, dataset="omni3d", sam2_predictor=None, device=None, trace_path=None
    ):
        super().__init__("same_object", trace_path)
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device

    def _get_bbox(self, mask, margin=20):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add margin
        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(mask.shape[0] - 1, rmax + margin)
        cmax = min(mask.shape[1] - 1, cmax + margin)

        return [cmin, rmin, cmax, rmax]

    def get_iou(self, box1, box2):
        # Coordinates of the intersection rectangle
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # Width and height of the intersection rectangle
        width_inter = max(0, x2_inter - x1_inter)
        height_inter = max(0, y2_inter - y1_inter)

        # Area of the intersection
        area_inter = width_inter * height_inter

        # Area of both bounding boxes
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Union of both bounding boxes
        area_union = area_box1 + area_box2 - area_inter

        # IoU calculation
        iou = area_inter / area_union if area_union != 0 else 0

        return iou

    def _get_mask(self, sam_inpt_pts, sam_inpt_label):
        with torch.no_grad():
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=sam_inpt_pts,
                point_labels=sam_inpt_label,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]

        return masks[0]

    def execute_bboxs(self, image, bbox1, bbox2):
        answer = self.get_iou(bbox1, bbox2) > 0.92
        boxed_image = box_image(image, [bbox1, bbox2])
        im_html = html_embed_image(boxed_image, 300)
        self.write_trace(im_html)
        self.write_trace(f"<p>{answer}<p>")

        return answer

    def execute_pts(self, image, x_1, y_1, x_2, y_2):
        if self.oracle:
            answer = self.oracle.same_object(x_1, y_1, x_2, y_2, self.scene_json)
        else:
            sam_inpt_label = np.array([1])  # foreground label
            obj_1_sam_inpt_pts = np.array([[int(x_1), int(y_1)]])
            obj_2_sam_inpt_pts = np.array([[int(x_2), int(y_2)]])
            self.sam2_predictor.set_image(np.array(image))

            obj_1_mask = self._get_mask(obj_1_sam_inpt_pts, sam_inpt_label)
            obj_2_mask = self._get_mask(obj_2_sam_inpt_pts, sam_inpt_label)

            obj_1_bbox = self._get_bbox(obj_1_mask)
            obj_2_bbox = self._get_bbox(obj_2_mask)

            answer = self.get_iou(obj_1_bbox, obj_2_bbox) > 0.92

        if self.oracle:
            self.write_trace(f"<p>Same Object [Oracle]: ({x_1}, {y_1}) and ({x_2}, {y_2})<p>")
        else:
            self.write_trace(f"<p>Same Object: ({x_1}, {y_1}) and ({x_2}, {y_2})<p>")
        boxed_image = box_image(image, [obj_1_bbox, obj_2_bbox])
        im_html = html_embed_image(boxed_image, 300)
        self.write_trace(im_html)
        self.write_trace(f"<p>Answer: {answer}<p>")

        return answer


class Get2DObjectSize(PredefinedModule):
    def __init__(
        self, dataset="omni3d", sam2_predictor=None, device=None, trace_path=None
    ):
        super().__init__("get_2D_object_size", trace_path)
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device

    def execute_bboxs(self, image, bbox):
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])

        # trace
        boxed_image = box_image(image, [bbox])
        boxed_im_html = html_embed_image(boxed_image, 300)
        self.write_trace(boxed_im_html)
        self.write_trace(f"<p>Width: {width}, Height: {height}<p>")

        return width, height

    def execute_pts(self, image, x, y):
        with torch.no_grad():
            sam_inpt_pts = np.array([[int(x), int(y)]])
            sam_inpt_label = np.array([1])  # foreground label
            self.sam2_predictor.set_image(np.array(image))

            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=sam_inpt_pts,
                point_labels=sam_inpt_label,
                multimask_output=True,
            )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        if scores[1] > 0.3:
            box1 = self._get_bbox(masks[0])
            box2 = self._get_bbox(masks[1])
            box = [
                min(box1[0], box2[0]),
                min(box1[1], box2[1]),
                max(box1[2], box2[2]),
                max(box1[3], box2[3]),
            ]
        elif scores[2] > 0.2:
            box1 = self._get_bbox(masks[0])
            box2 = self._get_bbox(masks[1])
            box3 = self._get_bbox(masks[2])
            box = [
                min(box1[0], box2[0], box3[0]),
                min(box1[1], box2[1], box3[1]),
                max(box1[2], box2[2], box3[2]),
                max(box1[3], box2[3], box3[3]),
            ]
        else:
            box = self._get_bbox(masks[0])

        width = abs(box[0] - box[2])
        height = abs(box[1] - box[3])

        # trace
        if self.oracle:
            self.write_trace(f"<p>Get 2D Object Size [Oracle]: ({x}, {y})<p>")
        else:
            self.write_trace(f"<p>Get 2D Object Size: ({x}, {y})<p>")
        boxed_image = box_image(image, [box])
        boxed_im_html = html_embed_image(boxed_image, 300)
        self.write_trace(boxed_im_html)
        self.write_trace(f"<p>Width: {width}, Height: {height}<p>")

        return width, height


class ResultModule(PredefinedModule):
    def __init__(self, trace_path=None):
        super().__init__("result", trace_path)

    def execute_pts(self, var):
        self.write_trace(f"<p>Result: {var}<p>")
        return str(var)

    def execute_bboxs(self, var):
        self.write_trace(f"<p>Result: {var}<p>")
        return str(var)


class ModulesList:
    def __init__(self, models_path=None, trace_path=None, dataset="omni3d", api_key_path="./api.key"):
        set_devices()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

        if dataset in ["clevr", "gqa"]:
            self.sam2_checkpoint = f"{models_path}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
            self.sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            self.sam2_predictor = SAM2ImagePredictor(
                build_sam2(
                    self.sam2_model_cfg, self.sam2_checkpoint, device=self.device
                )
            )
            print("SAM2 Initialized")
            self.molmo_processor = AutoProcessor.from_pretrained(
                "allenai/Molmo-7B-D-0924",
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            self.molmo_model = AutoModelForCausalLM.from_pretrained(
                "allenai/Molmo-7B-D-0924",
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            print("Molmo Initialized")
        else:
            self.grounding_dino = load_model(
                f"{models_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                f"{models_path}/GroundingDINO/weights/groundingdino_swint_ogc.pth"
            )
            print("GroundingDINO Initialized")

        self.unidepth_model = UniDepthV2.from_pretrained(
            "lpiccinelli/unidepth-v2-vits14"
        ).to(self.device)

        self.modules = self.get_module_list(self.dataset, trace_path, api_key_path)
        self.module_names = [module.name for module in self.modules]
        self.module_executes = self.get_module_executes(self.dataset)

    def get_module_executes(self, dataset):
        if dataset in ["clevr", "gqa"]:
            return {
                self.module_names[i]: self.modules[i].execute_pts
                for i in range(len(self.modules))
            }
        else:
            return {
                self.module_names[i]: self.modules[i].execute_bboxs
                for i in range(len(self.modules))
            }

    def get_module_list(self, dataset, trace_path, api_key_path):
        if dataset in ["clevr", "gqa"]:
            return [
                LocateModule(
                    dataset=dataset,
                    molmo_processor=self.molmo_processor,
                    molmo_model=self.molmo_model,
                    trace_path=trace_path,
                ),
                VQAModule(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                    api_key_path=api_key_path,
                ),
                DepthModule(self.unidepth_model, self.device, trace_path),
                SameObjectModule(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                ),
                Get2DObjectSize(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                ),
                ResultModule(trace_path),
            ]
        else:
            return [
                LocateModule(
                    dataset=dataset,
                    grounding_dino=self.grounding_dino,
                    trace_path=trace_path,
                ),
                VQAModule(
                    dataset=dataset, trace_path=trace_path, api_key_path=api_key_path
                ),
                DepthModule(self.unidepth_model, self.device, trace_path),
                SameObjectModule(dataset=dataset, trace_path=trace_path),
                Get2DObjectSize(dataset=dataset, trace_path=trace_path),
                ResultModule(trace_path),
            ]

    def set_trace_path(self, trace_path):
        for module in self.modules:
            module.trace_path = trace_path

    def set_oracle(self, oracle, reference_image, scene_json):
        for module in self.modules:
            if hasattr(module, "set_oracle"):
                module.set_oracle(oracle, reference_image, scene_json)

    def clear_oracle(self):
        for module in self.modules:
            if hasattr(module, "set_oracle"):
                module.clear_oracle()


# =============================================================================
# VIDEO MODE MODULES
# =============================================================================

class VideoTrackModule(PredefinedModule):
    """
    Track an object across video frames using SAM 2 Video Predictor.

    Given an initial point or bbox on a specific frame, propagates the mask
    forward and backward through the video, returning per-frame bounding boxes.
    """

    def __init__(self, sam2_video_predictor, device, chunk_size=None, trace_path=None):
        super().__init__("track", trace_path)
        self.predictor = sam2_video_predictor
        self.device = device
        self.chunk_size = chunk_size  # None = process all frames at once

    def execute(self, video_dir, init_frame_idx, x, y):
        """
        Track an object from an initial point across all frames.

        Args:
            video_dir (str): Path to directory of extracted JPEG frames.
            init_frame_idx (int): Frame index where the object is identified.
            x (int): X coordinate of the object in the initial frame.
            y (int): Y coordinate of the object in the initial frame.

        Returns:
            dict: {frame_idx (int): [x1, y1, x2, y2]} bounding boxes per frame.
        """
        from .video_utils import bbox_from_mask, get_frame_count, chunk_frame_indices

        total_frames = get_frame_count(video_dir)
        all_bboxes = {}

        if self.chunk_size and total_frames > self.chunk_size:
            chunks = chunk_frame_indices(total_frames, self.chunk_size)
        else:
            chunks = [(0, total_frames - 1)]

        for chunk_start, chunk_end in chunks:
            # Only process chunks that contain our init frame or are after it
            # SAM2 propagates forward from the init frame
            if chunk_end < init_frame_idx:
                continue

            # Adjust init frame index for this chunk
            local_init = max(0, init_frame_idx - chunk_start)

            with torch.inference_mode():
                state = self.predictor.init_state(video_path=video_dir)

                # Add the point prompt on the init frame
                frame_idx_in_video = init_frame_idx
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx_in_video,
                    obj_id=1,
                    points=np.array([[x, y]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )

                # Propagate through all frames
                for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                    mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                    bbox = bbox_from_mask(mask)
                    if bbox is not None:
                        all_bboxes[frame_idx] = bbox

                self.predictor.reset_state(state)

        # Trace
        self.write_trace(f"<p>Track object from frame {init_frame_idx} at ({x}, {y})</p>")
        self.write_trace(f"<p>Tracked across {len(all_bboxes)} of {total_frames} frames</p>")

        return all_bboxes

    def execute_with_bbox(self, video_dir, init_frame_idx, bbox):
        """
        Track an object from an initial bounding box across all frames.

        Args:
            video_dir (str): Path to directory of extracted JPEG frames.
            init_frame_idx (int): Frame index where the object is identified.
            bbox (list): Bounding box [x1, y1, x2, y2] of the object.

        Returns:
            dict: {frame_idx (int): [x1, y1, x2, y2]} bounding boxes per frame.
        """
        from .video_utils import bbox_from_mask, get_frame_count

        total_frames = get_frame_count(video_dir)
        all_bboxes = {}

        with torch.inference_mode():
            state = self.predictor.init_state(video_path=video_dir)

            # Add the box prompt on the init frame
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=init_frame_idx,
                obj_id=1,
                box=np.array(bbox, dtype=np.float32),
            )

            # Propagate through all frames
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                bbox_result = bbox_from_mask(mask)
                if bbox_result is not None:
                    all_bboxes[frame_idx] = bbox_result

            self.predictor.reset_state(state)

        # Trace
        self.write_trace(f"<p>Track object from frame {init_frame_idx} with bbox {bbox}</p>")
        self.write_trace(f"<p>Tracked across {len(all_bboxes)} of {total_frames} frames</p>")

        return all_bboxes


class VideoLocateModule(PredefinedModule):
    """
    Locate objects in a specific video frame.

    Delegates to the existing LocateModule by loading the requested frame
    and running detection on it.
    """

    def __init__(self, locate_module, trace_path=None):
        super().__init__("loc_frame", trace_path)
        self.locate_module = locate_module

    def execute(self, video_dir, frame_index, object_prompt):
        """
        Locate objects in a specific frame of the video.

        Args:
            video_dir (str): Path to directory of extracted JPEG frames.
            frame_index (int): Frame index to search.
            object_prompt (str): Description of object to locate.

        Returns:
            list: Bounding boxes [xmin, ymin, xmax, ymax] for located objects.
        """
        from .video_utils import load_frame

        frame = load_frame(video_dir, frame_index)

        # Delegate to the existing locate module's bboxs method
        if hasattr(self.locate_module, 'execute_bboxs'):
            result = self.locate_module.execute_bboxs(frame, object_prompt)
        else:
            result = self.locate_module.execute_pts(frame, object_prompt)

        self.write_trace(f"<p>Locate in frame {frame_index}: {object_prompt}</p>")
        self.write_trace(f"<p>Found {len(result)} objects</p>")

        return result


class VideoVQAModule(PredefinedModule):
    """
    Answer visual questions about objects in a specific video frame.

    Delegates to the existing VQAModule by loading the requested frame.
    """

    def __init__(self, vqa_module, trace_path=None):
        super().__init__("vqa_frame", trace_path)
        self.vqa_module = vqa_module

    def execute(self, video_dir, frame_index, question, bbox=None):
        """
        Answer a question about an object in a specific video frame.

        Args:
            video_dir (str): Path to directory of extracted JPEG frames.
            frame_index (int): Frame index to analyze.
            question (str): Question about the object.
            bbox (list, optional): Bounding box [xmin, ymin, xmax, ymax].

        Returns:
            str: Answer to the question.
        """
        from .video_utils import load_frame

        frame = load_frame(video_dir, frame_index)

        if bbox is not None:
            if hasattr(self.vqa_module, 'execute_bboxs'):
                answer = self.vqa_module.execute_bboxs(frame, question, bbox)
            else:
                # Use center of bbox as point
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                answer = self.vqa_module.execute_pts(frame, question, cx, cy)
        else:
            answer = self.vqa_module.predict(frame, question, holistic=True)

        self.write_trace(f"<p>VQA on frame {frame_index}: {question}</p>")
        self.write_trace(f"<p>Answer: {answer}</p>")

        return answer


class VideoGetFrameModule(PredefinedModule):
    """
    Load a specific frame from the extracted video frames directory.
    """

    def __init__(self, trace_path=None):
        super().__init__("get_frame", trace_path)

    def execute(self, video_dir, frame_index):
        """
        Load a frame as a PIL Image.

        Args:
            video_dir (str): Path to directory of extracted JPEG frames.
            frame_index (int): Zero-based index of the frame to load.

        Returns:
            PIL.Image: The frame as an RGB Image.
        """
        from .video_utils import load_frame

        frame = load_frame(video_dir, frame_index)
        self.write_trace(f"<p>Loaded frame {frame_index}</p>")
        return frame


class VideoResultModule(PredefinedModule):
    """
    Return the final result from a video analysis program.
    """

    def __init__(self, trace_path=None):
        super().__init__("result", trace_path)

    def execute(self, var):
        self.write_trace(f"<p>Result: {var}</p>")
        return str(var)


class VideoModulesList:
    """
    Initializes and manages video-mode modules.

    Uses SAM2VideoPredictor for object tracking across frames,
    alongside GroundingDINO/Molmo for per-frame detection and
    GPT-4o for visual question answering.
    """

    def __init__(self, models_path=None, trace_path=None, dataset="omni3d",
                 api_key_path="./api.key", chunk_size=None):
        """
        Args:
            models_path (str): Path to model weights directory.
            trace_path (str): Path for execution trace HTML file.
            dataset (str): Dataset type ('clevr', 'gqa', or 'omni3d').
            api_key_path (str): Path to OpenAI API key file.
            chunk_size (int, optional): Max frames to process at once.
                If None, auto-detects based on available VRAM.
        """
        set_devices()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

        # Auto-detect chunk size from VRAM if not specified
        if chunk_size is None:
            self.chunk_size = self._auto_chunk_size()
        else:
            self.chunk_size = chunk_size

        # Initialize SAM 2 in VIDEO mode
        self.sam2_checkpoint = f"{models_path}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        self.sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        self.sam2_video_predictor = SAM2VideoPredictor(
            build_sam2(self.sam2_model_cfg, self.sam2_checkpoint, device=self.device)
        )
        print("SAM2 Video Predictor Initialized")

        # Initialize object detection (same as image mode)
        if dataset in ["clevr", "gqa"]:
            self.molmo_processor = AutoProcessor.from_pretrained(
                "allenai/Molmo-7B-D-0924",
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            self.molmo_model = AutoModelForCausalLM.from_pretrained(
                "allenai/Molmo-7B-D-0924",
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            print("Molmo Initialized")
            self._locate_module = LocateModule(
                dataset=dataset,
                molmo_processor=self.molmo_processor,
                molmo_model=self.molmo_model,
                trace_path=trace_path,
            )
        else:
            from groundingdino.util.inference import load_model as load_gd_model
            self.grounding_dino = load_gd_model(
                f"{models_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                f"{models_path}/GroundingDINO/weights/groundingdino_swint_ogc.pth",
            )
            print("GroundingDINO Initialized")
            self._locate_module = LocateModule(
                dataset=dataset,
                grounding_dino=self.grounding_dino,
                trace_path=trace_path,
            )

        # Initialize VQA (reuse existing VQAModule)
        self._vqa_module = VQAModule(
            dataset=dataset,
            trace_path=trace_path,
            api_key_path=api_key_path,
        )

        # Build video module list
        self.modules = self._get_video_modules(trace_path)
        self.module_names = [module.name for module in self.modules]
        self.module_executes = {
            name: module.execute for name, module in zip(self.module_names, self.modules)
        }

    def _auto_chunk_size(self):
        """Auto-detect chunk size based on available GPU VRAM."""
        if not torch.cuda.is_available():
            return 20  # CPU: very conservative

        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        if vram_gb >= 24:
            return None  # No chunking needed
        elif vram_gb >= 16:
            return 100
        elif vram_gb >= 8:
            return 50
        else:
            return 20

    def _get_video_modules(self, trace_path):
        """Create the list of video-mode modules."""
        return [
            VideoTrackModule(
                sam2_video_predictor=self.sam2_video_predictor,
                device=self.device,
                chunk_size=self.chunk_size,
                trace_path=trace_path,
            ),
            VideoLocateModule(
                locate_module=self._locate_module,
                trace_path=trace_path,
            ),
            VideoVQAModule(
                vqa_module=self._vqa_module,
                trace_path=trace_path,
            ),
            VideoGetFrameModule(trace_path=trace_path),
            VideoResultModule(trace_path=trace_path),
        ]

    def set_trace_path(self, trace_path):
        for module in self.modules:
            module.trace_path = trace_path
