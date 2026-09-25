import torch
import torch.nn.functional as F


CANVAS_MULTIPLE = 32

QUALITY_OPTIONS = [
    "200",
    "256",
    "320",
    "350",
    "400",
    "448",
    "480",
    "500",
    "512",
    "576",
    "600",
    "640",
    "704",
    "720",
    "768",
    "832",
    "896",
    "960",
    "1024",
    "1080",
    "1152",
    "1280",
    "1440",
    "1600",
    "1920",
    "2048",
]

LEFT_QUALITY_OPTIONS = ["max"] + QUALITY_OPTIONS

ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}

SEAM_TRIM_OPTIONS = ["auto", "0", "8", "16", "24", "32", "48", "64", "96", "128"]


def _snap_nearest(value):
    return max(CANVAS_MULTIPLE, int(round(float(value) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def _snap_down(value):
    return max(CANVAS_MULTIPLE, int(float(value) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def _resize_frames(images, width, height):
    if images.shape[2] == width and images.shape[1] == height:
        return images

    resized = F.interpolate(
        images.movedim(-1, 1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _fit_source_dimensions(source_width, source_height, quality):
    source_short = min(source_width, source_height)
    requested = min(source_short, 2048 if quality == "max" else int(quality))

    if requested >= source_short:
        target_short = _snap_down(source_short)
    else:
        target_short = _snap_nearest(requested)
        if target_short > source_short:
            target_short = _snap_down(source_short)

    scale = target_short / source_short
    target_width = min(_snap_nearest(source_width * scale), _snap_down(source_width))
    target_height = min(_snap_nearest(source_height * scale), _snap_down(source_height))
    return target_width, target_height


def _dimensions_from_aspect(aspect_ratio, short_edge):
    ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
    short_edge = _snap_nearest(int(short_edge))

    if ratio_width <= ratio_height:
        width = short_edge
        height = _snap_nearest(short_edge * ratio_height / ratio_width)
    else:
        height = short_edge
        width = _snap_nearest(short_edge * ratio_width / ratio_height)

    return width, height


def _detect_left_black_band(images):
    """Find a persistent near-black band at the left edge without scanning every pixel."""
    frame_step = max(1, images.shape[0] // 16)
    row_step = max(1, images.shape[1] // 128)
    sample = images[::frame_step, ::row_step, :, :3].float()
    luminance = (
        sample[..., 0] * 0.2126
        + sample[..., 1] * 0.7152
        + sample[..., 2] * 0.0722
    )
    profile = torch.quantile(
        luminance.permute(2, 0, 1).reshape(images.shape[2], -1),
        0.95,
        dim=1,
    )

    max_trim = min(images.shape[2] // 4, 128)
    detected = 0
    for value in profile[:max_trim]:
        if value.item() <= 0.08:
            detected += 1
        else:
            break

    if detected < 4:
        return 0
    return min(detected + 2, max_trim)


class TSHalfMaskVideoLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE",),
                "left_quality": (LEFT_QUALITY_OPTIONS, {"default": "350"}),
                "right_aspect_ratio": (
                    list(ASPECT_RATIOS),
                    {"default": "9:16 (Portrait Widescreen)"},
                ),
                "right_quality": (QUALITY_OPTIONS, {"default": "350"}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "TS_HALF_MASK_LAYOUT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "wide_canvas",
        "right_noise_mask",
        "layout",
        "canvas_width",
        "canvas_height",
        "frame_count",
        "right_width",
        "right_height",
    )
    FUNCTION = "build"
    CATEGORY = "Teskor's Utils/Video"
    DESCRIPTION = (
        "Builds a side-by-side canvas for Half Mask Video. The LEFT source panel is "
        "preserved, the RIGHT panel is masked for generation, source resolution is "
        "never upscaled, and all dimensions are snapped to a 32-pixel grid."
    )

    def build(self, source_video, left_quality, right_aspect_ratio, right_quality):
        if source_video.ndim != 4 or source_video.shape[-1] not in (3, 4):
            raise ValueError(
                "source_video must be an IMAGE batch shaped [frames, height, width, channels]"
            )

        frame_count, source_height, source_width, channels = source_video.shape
        if frame_count < 1:
            raise ValueError("source_video contains no frames")

        left_width, left_height = _fit_source_dimensions(
            source_width,
            source_height,
            left_quality,
        )
        right_width, right_height = _dimensions_from_aspect(
            right_aspect_ratio,
            right_quality,
        )

        canvas_width = left_width + right_width
        canvas_height = max(left_height, right_height)
        left_y = (canvas_height - left_height) // 2
        right_x = left_width
        right_y = (canvas_height - right_height) // 2

        resized_source = _resize_frames(source_video, left_width, left_height)
        canvas = torch.zeros(
            (frame_count, canvas_height, canvas_width, channels),
            dtype=source_video.dtype,
            device=source_video.device,
        )
        canvas[:, left_y : left_y + left_height, :left_width, :] = resized_source

        noise_mask = torch.zeros(
            (frame_count, canvas_height, canvas_width),
            dtype=source_video.dtype,
            device=source_video.device,
        )
        noise_mask[
            :,
            right_y : right_y + right_height,
            right_x : right_x + right_width,
        ] = 1.0

        layout = {
            "version": 1,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "left_x": 0,
            "left_y": left_y,
            "left_width": left_width,
            "left_height": left_height,
            "right_x": right_x,
            "right_y": right_y,
            "right_width": right_width,
            "right_height": right_height,
            "frame_count": frame_count,
            "left_quality": left_quality,
            "right_quality": right_quality,
            "right_aspect_ratio": right_aspect_ratio,
        }

        return (
            canvas,
            noise_mask,
            layout,
            canvas_width,
            canvas_height,
            frame_count,
            right_width,
            right_height,
        )


class TSHalfMaskVideoExtractGenerated:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoded_wide_video": ("IMAGE",),
                "layout": ("TS_HALF_MASK_LAYOUT",),
                "left_seam_trim": (SEAM_TRIM_OPTIONS, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_video",)
    FUNCTION = "extract"
    CATEGORY = "Teskor's Utils/Video"
    DESCRIPTION = (
        "Returns only the generated RIGHT panel from a decoded Half Mask Video result."
    )

    def extract(self, decoded_wide_video, layout, left_seam_trim="auto"):
        if not isinstance(layout, dict) or layout.get("version") != 1:
            raise ValueError("Invalid Half Mask Video layout")
        if decoded_wide_video.ndim != 4:
            raise ValueError("decoded_wide_video must be an IMAGE batch")

        actual_height = decoded_wide_video.shape[1]
        actual_width = decoded_wide_video.shape[2]
        scale_x = actual_width / layout["canvas_width"]
        scale_y = actual_height / layout["canvas_height"]

        x = round(layout["right_x"] * scale_x)
        y = round(layout["right_y"] * scale_y)
        width = round(layout["right_width"] * scale_x)
        height = round(layout["right_height"] * scale_y)
        x = max(0, min(x, actual_width - 1))
        y = max(0, min(y, actual_height - 1))
        width = max(1, min(width, actual_width - x))
        height = max(1, min(height, actual_height - y))

        generated = decoded_wide_video[:, y : y + height, x : x + width, :]
        seam_trim = (
            _detect_left_black_band(generated)
            if left_seam_trim == "auto"
            else int(left_seam_trim)
        )
        seam_trim = max(0, min(seam_trim, generated.shape[2] - CANVAS_MULTIPLE))
        if seam_trim:
            generated = generated[:, :, seam_trim:, :]

        if (
            generated.shape[2] != layout["right_width"]
            or generated.shape[1] != layout["right_height"]
        ):
            generated = _resize_frames(
                generated,
                layout["right_width"],
                layout["right_height"],
            )

        return (generated,)
