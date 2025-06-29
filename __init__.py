import torch
import comfy.utils
from .Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace
from comfy.model_management import get_torch_device


class AutoCropFaces:
    def __init__(self):
        pass

    # ─────────────────────────────────────────────────────────
    #  INVOERVELDEN (1 NIEUWE: sort_by)
    # ─────────────────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "number_of_faces": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10,
                    "step": 0.5,
                    "display": "slider"
                }),
                "shift_factor": ("FLOAT", {
                    "default": 0.45,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "step": 1,
                    "display": "number"
                }),
                "max_faces_per_image": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                }),
                "aspect_ratio": (["9:16", "2:3", "3:4", "4:5", "1:1",
                                  "5:4", "4:3", "3:2", "16:9"], {
                    "default": "1:1",
                }),
                # ▼▼▼  NIEUW ▼▼▼
                "sort_by": ("STRING", {
                    "default": "confidence",
                    "choices": [
                        "confidence",      # detector-score (origineel)
                        "area_desc",       # groot → klein
                        "area_asc",        # klein → groot
                        "x_left2right"     # links → rechts
                    ],
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("face",)
    FUNCTION = "auto_crop_faces"
    CATEGORY = "Faces"

    # ---------------------------------------------------------
    #  HULPFUNCTIES
    # ---------------------------------------------------------
    def aspect_ratio_string_to_float(self, str_aspect_ratio="1:1"):
        a, b = map(float, str_aspect_ratio.split(':'))
        return a / b

    def auto_crop_faces_in_image(
        self,
        image,
        max_number_of_faces,
        scale_factor,
        shift_factor,
        aspect_ratio,
        method='lanczos'
    ):
        image_255 = image * 255
        rf = Pytorch_RetinaFace(top_k=50,
                                keep_top_k=max_number_of_faces,
                                device=get_torch_device())
        dets = rf.detect_faces(image_255)
        cropped_faces, bbox_info = rf.center_and_crop_rescale(
            image,
            dets,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
            aspect_ratio=aspect_ratio
        )
        # voeg batch-dimensie toe
        cropped_faces_with_batch = [face.unsqueeze(0) for face in cropped_faces]
        return cropped_faces_with_batch, bbox_info

    # ---------------------------------------------------------
    #  HOOFDFUNCTIE
    # ---------------------------------------------------------
    def auto_crop_faces(
        self,
        image,
        number_of_faces,
        start_index,
        max_faces_per_image,
        scale_factor,
        shift_factor,
        aspect_ratio,
        sort_by="confidence",          # ← nieuw argument
        method='lanczos'
    ):
        """
        image            – (batch, w, h, c)
        number_of_faces  – max. aantal croppen dat terugkomt
        start_index      – rotatiestart binnen de lijst
        sort_by          – confidence | area_desc | area_asc | x_left2right
        """
        # aspect-ratio string → float
        aspect_ratio = self.aspect_ratio_string_to_float(aspect_ratio)

        selected_faces, detected_cropped_faces = [], []
        selected_crop_data, detected_crop_data = [], []
        original_images = []

        # ----------- GEZICHTEN DETECTEREN PER BATCH ----------
        for i in range(image.shape[0]):
            original_images.append(image[i].unsqueeze(0))
            cropped_images, infos = self.auto_crop_faces_in_image(
                image[i],
                max_faces_per_image,
                scale_factor,
                shift_factor,
                aspect_ratio,
                method
            )
            detected_cropped_faces.extend(cropped_images)
            detected_crop_data.extend(infos)

        # niets gevonden → origineel terug
        if not detected_cropped_faces:
            selected_crop_data = [
                (0, 0, img.shape[3], img.shape[2]) for img in original_images
            ]
            return (image, selected_crop_data)

        # ------------- ▼  SORTEREN VOOR AFKAPPEN  ▼ -------------
        if sort_by != "confidence":
            pairs = list(zip(detected_cropped_faces, detected_crop_data))

            if sort_by in ("area_desc", "area_asc"):
                reverse = sort_by == "area_desc"
                key_fn = lambda p: p[0].shape[1] * p[0].shape[2]  # W*H
            elif sort_by == "x_left2right":
                reverse = False
                key_fn = lambda p: p[1][0]                       # x1
            else:
                key_fn = None

            if key_fn is not None:
                pairs.sort(key=key_fn, reverse=reverse)
                detected_cropped_faces, detected_crop_data = map(
                    list, zip(*pairs)
                )

        # ------------------ AFKAPPEN ---------------------------
        start_index = start_index % len(detected_cropped_faces)

        if number_of_faces >= len(detected_cropped_faces):
            selected_faces = (detected_cropped_faces[start_index:] +
                              detected_cropped_faces[:start_index])
            selected_crop_data = (detected_crop_data[start_index:] +
                                  detected_crop_data[:start_index])
        else:
            end_index = (start_index + number_of_faces) % len(detected_cropped_faces)
            if start_index < end_index:
                selected_faces = detected_cropped_faces[start_index:end_index]
                selected_crop_data = detected_crop_data[start_index:end_index]
            else:
                selected_faces = (detected_cropped_faces[start_index:] +
                                  detected_cropped_faces[:end_index])
                selected_crop_data = (detected_crop_data[start_index:] +
                                      detected_crop_data[:end_index])

        # fallback: niks geselecteerd
        if not selected_faces:
            return (image, None)

        # één gezicht → direct terug
        if len(selected_faces) == 1:
            out = selected_faces[0]
            crop_data = selected_crop_data[0]
            return (out, crop_data)

        # ---------- UNIFORME RESOLUTIE AF DIT PUNT -------------
        max_width_index = max(range(len(selected_faces)),
                              key=lambda i: selected_faces[i].shape[1])
        max_width = selected_faces[max_width_index].shape[1]
        max_height = selected_faces[max_width_index].shape[2]
        shape = (max_height, max_width)

        out = None
        for face_image in selected_faces:
            if shape != face_image.shape[1:3]:
                face_image = comfy.utils.common_upscale(
                    face_image.movedim(-1, 1),
                    max_height,
                    max_width,
                    method,
                    ""
                ).movedim(1, -1)
            out = face_image if out is None else torch.cat((out, face_image), dim=0)

        return (out, selected_crop_data)


# ─────────────────────────────────────────────────────────────
#  Registratie
# ─────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "AutoCropFaces": AutoCropFaces,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropFaces": "Auto Crop Faces",
}
