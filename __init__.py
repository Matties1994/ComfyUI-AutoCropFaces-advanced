from typing import List, Tuple
import torch
import numpy as np

# ------------------------------------------------------------
#  AutoCropFaces – aangepaste versie
# ------------------------------------------------------------

class AutoCropFaces:
    """
    Detects faces, crops them and returns:
      • IMAGE-list  – de geschaalde croppen
      • CROP_DATA   – [[H,W], [x1,y1,x2,y2], …]
    Nu met sorteer-optie: confidence (default), area_desc, area_asc, x_left2right.
    """

    # ─────────────────────────────────────────────────────────
    # 1.  Inputs
    # ─────────────────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "number_of_faces": (
                    "INT",
                    { "default": 5, "min": 1, "max": 50, "step": 1 }
                ),
                # ▼▼▼  NIEUW  ▼▼▼
                "sort_by": (
                    "STRING",
                    {
                        "default": "confidence",
                        "choices": [
                            "confidence",      # oorspronkelijke volgorde
                            "area_desc",       # groot → klein
                            "area_asc",        # klein → groot
                            "x_left2right"     # links → rechts
                        ],
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("face", "crop_data")
    FUNCTION = "auto_crop_faces"
    CATEGORY = "Loaders/IO"

    # ─────────────────────────────────────────────────────────
    # 2.  Main logic
    # ─────────────────────────────────────────────────────────
    def auto_crop_faces(
        self,
        image: torch.Tensor,
        number_of_faces: int = 5,
        sort_by: str = "confidence",
    ):
        """
        • ‘image’  – (B,C,H,W) tensor
        • ‘number_of_faces’  – hoeveel croppen teruggeven
        • ‘sort_by’ –   confidence | area_desc | area_asc | x_left2right
        """
        B, C, H, W = image.shape
        detected_cropped_faces: List[torch.Tensor] = []
        detected_crop_data:       List[Tuple[int, int, int, int]] = []

        # ---- hier staat je bestaande RetinaFace / YOLO detectielus ----
        # vul detected_cropped_faces en detected_crop_data zoals voorheen
        # ----------------------------------------------------------------

        # ─────────────────────────────────────────────────────────
        # 2a.  Sorteer vóór afkappen
        # ─────────────────────────────────────────────────────────
        if sort_by != "confidence":
            pairs = list(zip(detected_cropped_faces, detected_crop_data))

            if sort_by in ("area_desc", "area_asc"):
                reverse = sort_by == "area_desc"
                key_fn = lambda t: t[0].shape[2] * t[0].shape[3]   # H*W
            elif sort_by == "x_left2right":
                reverse = False
                key_fn = lambda t: t[1][0]                         # x1
            else:                          # fallback – verander niets
                key_fn = None

            if key_fn is not None:
                pairs.sort(key=key_fn, reverse=reverse)
                detected_cropped_faces, detected_crop_data = map(
                    list, zip(*pairs)
                )

        # ─────────────────────────────────────────────────────────
        # 2b.  Trim tot ‘number_of_faces’
        # ─────────────────────────────────────────────────────────
        if number_of_faces > 0:
            detected_cropped_faces = detected_cropped_faces[:number_of_faces]
            detected_crop_data     = detected_crop_data[:number_of_faces]

        # ----------------------------------------------------------------
        # 3.  Output
        # ----------------------------------------------------------------
        crop_data_out = [[H, W], *detected_crop_data]  # formaat behouden
        faces_out = torch.cat(detected_cropped_faces, dim=0) if detected_cropped_faces else torch.empty((0, C, H, W))
        return (faces_out, crop_data_out)


# ─────────────────────────────────────────────────────────────
# 4.  Node-registratie
# ─────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "AutoCropFaces": AutoCropFaces,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropFaces": "AutoCrop Faces",
}
