import base64
import io
from dataclasses import dataclass

from PIL import Image as PILImage

from ..core.types import ImageContent


@dataclass(frozen=True)
class ImageDimensions:
    width: int
    height: int


def image_content_to_pil(image: ImageContent) -> PILImage.Image:
    data = base64.b64decode(image.data)
    return PILImage.open(io.BytesIO(data))


def get_image_dimensions(image: ImageContent) -> ImageDimensions | None:
    try:
        with image_content_to_pil(image) as img:
            width, height = img.size
        return ImageDimensions(width=width, height=height)
    except Exception:
        return None


def image_fallback(image: ImageContent) -> str:
    dimensions = get_image_dimensions(image)
    if dimensions is None:
        return f"[Image: {image.mime_type}]"
    return f"[Image: {image.mime_type} {dimensions.width}x{dimensions.height}]"
