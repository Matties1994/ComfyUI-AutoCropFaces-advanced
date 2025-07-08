import torch
import torchvision.ops as ops  # NMS voor duplicate filtering
import comfy.utils
from .Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace
from comfy.model_management import get_torch_device


class AutoCropFaces:
    """Detecteert gezichten, filtert duplicaten met NMS, sorteert en crop.
    
    *Nieuw*
    --------
    - **NMS** (Non‑Maximum Suppression) in `_detect_and_crop` voorkomt dat hetzelfde
      gezicht meerdere keren wordt doorgegeven.
    - Parameter `face_filter` bepaalt volgorde (grootste, kleinste, links‑naar‑rechts).
    """

    # ───────────────────────────────────────────────────────────── INPUT TYPES ──
    @classmethod
    def INPUT_TYPES(cls):
        """Definieert gui‑inputs, incl. dropdown 'face_filter'."""
        return {
            "required": {
                "image": ("IMAGE",),

                # number of faces to output
                "number_of_faces": (
                    "INT",
                    {"default": 5, "min": 1, "max": 100, "step": 1},
                ),

                # detection / crop parameters
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.5,
                        "max": 20,
                        "step": 0.5,
                        "display": "slider",
                    },
                ),
                "shift_factor": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),

                # start position in ordered list (circular)
                "start_index": ("INT", {"default": 0, "step": 1, "display": "number"}),

                # max faces to *detect* per image
                "max_faces_per_image": (
                    "INT",
                    {"default": 50, "min": 1, "max": 1000, "step": 1},
                ),

                # output aspect‑ratio
                "aspect_ratio": (
                    [
                        "9:16",
                        "2:3",
                        "3:4",
                        "4:5",
                        "1:1",
                        "5:4",
                        "4:3",
                        "3:2",
                        "16:9",
                    ],
                    {"default": "1:1"},
                ),

                # ── NEW ── how to sort faces before selection
                "face_filter": (
                    ["largest_first", "smallest_first", "left_to_right"],
                    {"default": "largest_first"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("face",)
    FUNCTION = "auto_crop_faces"
    CATEGORY = "Faces"

    # ───────────────────────────────────────────────────────────── HELPERS ──
    @staticmethod
    def _aspect_ratio_to_float(ratio_str: str) -> float:
        a, b = map(float, ratio_str.split(":"))
        return a / b

    def _detect_and_crop(
        self,
        img_tensor: torch.Tensor,
        max_faces: int,
        scale: float,
        shift: float,
        aspect_ratio_float: float,
        method: str = "lanczos",
        nms_iou: float = 0.50,
    ):
        """Run RetinaFace, filter duplicaten met NMS en crop.

        Retourneert een tuple:
        - list[cropped_face_tensor]
        - list[tuple]  (cx, cy, scale)
        """
        # RetinaFace verwacht [0‑255] range
        img_255 = img_tensor * 255
        rf = Pytorch_RetinaFace(top_k=50, keep_top_k=max_faces, device=get_torch_device())

        # ── 1. Detectie ────────────────────────────────────────────────
        detections = rf.detect_faces(img_255)  # Tensor (N, 15)

        # ── 2. Duplicate‑filter met NMS ───────────────────────────────
        if detections.numel():  # alleen als er iets gedetecteerd is
            boxes = detections[:, :4]   # (x1, y1, x2, y2)
            scores = detections[:, 4]   # confidence
            keep = ops.nms(boxes, scores, nms_iou)
            detections = detections[keep]

        # ── 3. Croppen & rescalen ─────────────────────────────────────
        crops, infos = rf.center_and_crop_rescale(
            img_tensor,
            detections,
            scale_factor=scale,
            shift_factor=shift,
            aspect_ratio=aspect_ratio_float,
        )

        # Zorg dat batch‑dim aanwezig is
        crops = [c.unsqueeze(0) for c in crops]
        return crops, infos

    # ────────────────────────────────────────────────────────────── MAIN ──
    def auto_crop_faces(
        self,
        image: torch.Tensor,
        number_of_faces: int,
        start_index: int,
        max_faces_per_image: int,
        scale_factor: float,
        shift_factor: float,
        aspect_ratio: str,
        face_filter: str,  # new parameter
        method: str = "lanczos",
    ):
        """Detect faces, order via *face_filter*, dedup via NMS, return subset."""

        aspect_ratio_f = self._aspect_ratio_to_float(aspect_ratio)

        detected_faces, detected_infos = [], []

        # ── Detectie per batch‑frame ───────────────────────────────────
        for i in range(image.shape[0]):
            crops, infos = self._detect_and_crop(
                image[i],
                max_faces_per_image,
                scale_factor,
                shift_factor,
                aspect_ratio_f,
                method,
            )
            detected_faces.extend(crops)
            detected_infos.extend(infos)

        # ── Niets gevonden → passthrough ──────────────────────────────
        if not detected_faces:
            fallback = [(0, 0, image.shape[3], image.shape[2])]  # dummy crop
            return image, fallback

        # ── ORDER / FILTER ────────────────────────────────────────────
        if face_filter == "largest_first":
            order = sorted(
                range(len(detected_faces)),
                key=lambda i: detected_faces[i].shape[1] * detected_faces[i].shape[2],
                reverse=True,
            )
        elif face_filter == "smallest_first":
            order = sorted(
                range(len(detected_faces)),
                key=lambda i: detected_faces[i].shape[1] * detected_faces[i].shape[2],
            )
        else:  # "left_to_right"
            order = sorted(range(len(detected_faces)), key=lambda i: detected_infos[i][0])

        detected_faces = [detected_faces[i] for i in order]
        detected_infos = [detected_infos[i] for i in order]

        # ── SELECT SUBSET (circular slice) ────────────────────────────
        start_index %= len(detected_faces)

        if number_of_faces >= len(detected_faces):
            faces_sel = detected_faces[start_index:] + detected_faces[:start_index]
            infos_sel = detected_infos[start_index:] + detected_infos[:start_index]
        else:
            end = (start_index + number_of_faces) % len(detected_faces)
            if start_index < end:
                faces_sel = detected_faces[start_index:end]
                infos_sel = detected_infos[start_index:end]
            else:
                faces_sel = detected_faces[start_index:] + detected_faces[:end]
                infos_sel = detected_infos[start_index:] + detected_infos[:end]

        # ── PREPARE OUTPUT ───────────────────────────────────────────
        if not faces_sel:
            return image, None
        if len(faces_sel) == 1:
            return faces_sel[0], infos_sel[0]

        # Pad / upscale naar gelijke resolutie
        max_w = max(f.shape[1] for f in faces_sel)
        max_h = max(f.shape[2] for f in faces_sel)

        out = None
        for f in faces_sel:
            if (f.shape[1], f.shape[2]) != (max_w, max_h):
                f = comfy.utils.common_upscale(
                    f.movedim(-1, 1),  # (B,C,H,W)
                    max_h,
                    max_w,
                    method,
                    "",
                ).movedim(1, -1)
            out = f if out is None else torch.cat((out, f), dim=0)

        return out, infos_sel


# ────────────────────────────────────────────────────────── NODE REGISTRY ──
NODE_CLASS_MAPPINGS = {"AutoCropFaces": AutoCropFaces}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoCropFaces": "Auto Crop Faces"}
