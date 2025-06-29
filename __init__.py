import torch
import comfy.utils
from .Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace
from comfy.model_management import get_torch_device


# ─────────────────────────────────────────────────────────────
#  Auto Crop Faces – custom variant met sort_by-optie
# ─────────────────────────────────────────────────────────────
class AutoCropFacesCustom:           # ← unieke klasse-naam
    def __init__(self):
        pass

    # ---------------------------------------------------------
    #  INPUT-velden (extra: sort_by)
    # ---------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "number_of_faces": ("INT", {
                    "default": 5, "min": 1, "max": 100, "step": 1}),
                "scale_factor": ("FLOAT", {
                    "default": 1.5, "min": 0.5, "max": 10,
                    "step": 0.5, "display": "slider"}),
                "shift_factor": ("FLOAT", {
                    "default": 0.45, "min": 0, "max": 1,
                    "step": 0.01, "display": "slider"}),
                "start_index": ("INT", {"default": 0, "step": 1,
                                        "display": "number"}),
                "max_faces_per_image": ("INT", {
                    "default": 50, "min": 1, "max": 1000, "step": 1}),
                "aspect_ratio": (["9:16", "2:3", "3:4", "4:5", "1:1",
                                  "5:4", "4:3", "3:2", "16:9"], {
                    "default": "1:1"}),
                # ▼▼▼ NIEUW ▼▼▼
                "sort_by": ("STRING", {
                    "default": "confidence",
                    "choices": ["confidence",    # origineel
                                "area_desc",     # groot → klein
                                "area_asc",      # klein → groot
                                "x_left2right"]  # links → rechts
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("face",)
    FUNCTION = "auto_crop_faces"
    CATEGORY = "Faces"

    # ---------------------------------------------------------
    #  HULP
    # ---------------------------------------------------------
    def aspect_ratio_string_to_float(self, s="1:1"):
        a, b = map(float, s.split(':'))
        return a / b

    def auto_crop_faces_in_image(
        self, image, max_number_of_faces,
        scale_factor, shift_factor, aspect_ratio,
        method='lanczos'
    ):
        image_255 = image * 255
        rf = Pytorch_RetinaFace(top_k=50, keep_top_k=max_number_of_faces,
                                device=get_torch_device())
        dets = rf.detect_faces(image_255)
        cropped_faces, bbox_info = rf.center_and_crop_rescale(
            image, dets,
            scale_factor=scale_factor,
            shift_factor=shift_factor,
            aspect_ratio=aspect_ratio
        )
        return [f.unsqueeze(0) for f in cropped_faces], bbox_info

    # ---------------------------------------------------------
    #  MAIN
    # ---------------------------------------------------------
    def auto_crop_faces(
        self, image,
        number_of_faces, start_index, max_faces_per_image,
        scale_factor, shift_factor, aspect_ratio,
        sort_by="confidence", method='lanczos'
    ):
        aspect_ratio = self.aspect_ratio_string_to_float(aspect_ratio)

        detected_faces, detected_data = [], []
        originals = []

        for i in range(image.shape[0]):
            originals.append(image[i].unsqueeze(0))
            faces, infos = self.auto_crop_faces_in_image(
                image[i], max_faces_per_image,
                scale_factor, shift_factor, aspect_ratio, method)
            detected_faces.extend(faces)
            detected_data.extend(infos)

        # niets gevonden
        if not detected_faces:
            crop_default = [(0, 0, img.shape[3], img.shape[2]) for img in originals]
            return image, crop_default

        # ---------- sorteer vóór afkappen ----------
        if sort_by != "confidence":
            pairs = list(zip(detected_faces, detected_data))
            if sort_by in ("area_desc", "area_asc"):
                rev = sort_by == "area_desc"
                key = lambda p: p[0].shape[1] * p[0].shape[2]
            elif sort_by == "x_left2right":
                rev = False
                key = lambda p: p[1][0]            # x1
            else:
                key = None
            if key:
                pairs.sort(key=key, reverse=rev)
                detected_faces, detected_data = map(list, zip(*pairs))

        # ---------- afkappen ----------
        start_index %= len(detected_faces)
        if number_of_faces >= len(detected_faces):
            sel_faces = detected_faces[start_index:] + detected_faces[:start_index]
            sel_data  = detected_data[start_index:] + detected_data[:start_index]
        else:
            end = (start_index + number_of_faces) % len(detected_faces)
            if start_index < end:
                sel_faces = detected_faces[start_index:end]
                sel_data  = detected_data[start_index:end]
            else:
                sel_faces = detected_faces[start_index:] + detected_faces[:end]
                sel_data  = detected_data[start_index:] + detected_data[:end]

        if not sel_faces:
            return image, None
        if len(sel_faces) == 1:
            return sel_faces[0], sel_data[0]

        # uniforme resolutie
        idx_max = max(range(len(sel_faces)), key=lambda i: sel_faces[i].shape[1])
        mw = sel_faces[idx_max].shape[1]
        mh = sel_faces[idx_max].shape[2]
        out = None
        for f in sel_faces:
            if (mh, mw) != f.shape[1:3]:
                f = comfy.utils.common_upscale(
                    f.movedim(-1, 1), mh, mw, method, "").movedim(1, -1)
            out = f if out is None else torch.cat((out, f), dim=0)

        return out, sel_data


# ─────────────────────────────────────────────────────────────
#  Registratie
# ─────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "AutoCropFacesCustom": AutoCropFacesCustom   # ← mapping-naam
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropFacesCustom": "Auto Crop Faces (custom)"
}
