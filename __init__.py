import torch
import comfy.utils
from .Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace
from comfy.model_management import get_torch_device


class AutoCropFaces:
    # ───────────────────────────────────────────────────────────── INPUT TYPES ──
    @classmethod
    def INPUT_TYPES(cls):
        """Add dropdown 'face_filter' to choose sort order."""
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
                        "max": 10,
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

                # output aspect-ratio
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
        img_tensor,
        max_faces,
        scale,
        shift,
        aspect_ratio_float,
        method="lanczos",
    ):
        """
        Run RetinaFace on a single image tensor and return:
        • list[cropped_face_tensor]
        • list[tuple]  (cx, cy, scale) — centre-X, centre-Y, size factor
        """
        img_255 = img_tensor * 255
        rf = Pytorch_RetinaFace(
            top_k=50, keep_top_k=max_faces, device=get_torch_device()
        )
        detections = rf.detect_faces(img_255)
        crops, infos = rf.center_and_crop_rescale(
            img_tensor,
            detections,
            scale_factor=scale,
            shift_factor=shift,
            aspect_ratio=aspect_ratio_float,
        )
        # ensure batch-dim
        crops = [c.unsqueeze(0) for c in crops]
        return crops, infos

    # ────────────────────────────────────────────────────────────── MAIN ──
    def auto_crop_faces(
        self,
        image,
        number_of_faces,
        start_index,
        max_faces_per_image,
        scale_factor,
        shift_factor,
        aspect_ratio,
        face_filter,  # new parameter
        method="lanczos",
    ):
        """
        Detect faces, order them according to *face_filter*,
        then return `number_of_faces` starting at `start_index`.
        """

        aspect_ratio_f = self._aspect_ratio_to_float(aspect_ratio)

        detected_faces, detected_infos = [], []

        # iterate over batch
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

        # nothing detected → pass-through
        if not detected_faces:
            fallback = [(0, 0, image.shape[3], image.shape[2])]  # dummy crop
            return image, fallback

        # ── ORDER / FILTER ───────────────────────────────────────────────
        if face_filter == "largest_first":
            order = sorted(
                range(len(detected_faces)),
                key=lambda i: detected_faces[i].shape[1]  # width
                * detected_faces[i].shape[2],  # height
                reverse=True,
            )
        elif face_filter == "smallest_first":
            order = sorted(
                range(len(detected_faces)),
                key=lambda i: detected_faces[i].shape[1]
                * detected_faces[i].shape[2],
            )
        else:  # "left_to_right"
            # infos[0] is centre-x for every face tuple
            order = sorted(range(len(detected_faces)), key=lambda i: detected_infos[i][0])

        detected_faces = [detected_faces[i] for i in order]
        detected_infos = [detected_infos[i] for i in order]

        # ── SELECT SUBSET (circular slice) ───────────────────────────────
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

        # ── PREPARE OUTPUT ───────────────────────────────────────────────
        if not faces_sel:
            return image, None
        if len(faces_sel) == 1:
            return faces_sel[0], infos_sel[0]

        # pad / upscale to common size
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
