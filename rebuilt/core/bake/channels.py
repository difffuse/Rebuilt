"""Principled material channels shared by baking, suggestions and the UI."""

EXTRA_CHANNELS = {
    "SPECULAR": ("Specular", "Specular IOR Level", False, 0.5),
    "SPECULAR_TINT": ("Specular Tint", "Specular Tint", True, (1.0, 1.0, 1.0, 1.0)),
    "IOR": ("IOR", "IOR", False, 1.5),
    "EMISSION": ("Emission", "Emission Color", True, (0.0, 0.0, 0.0, 1.0)),
    "EMISSION_STRENGTH": ("Emission Strength", "Emission Strength", False, 0.0),
    "TRANSMISSION": ("Transmission", "Transmission Weight", False, 0.0),
    "COAT": ("Coat", "Coat Weight", False, 0.0),
    "COAT_ROUGHNESS": ("Coat Roughness", "Coat Roughness", False, 0.03),
    "SUBSURFACE": ("Subsurface", "Subsurface Weight", False, 0.0),
    "SUBSURFACE_SCALE": ("Subsurface Scale", "Subsurface Scale", False, 0.005),
}

IMAGE_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "TIFF": "tif", "OPEN_EXR": "exr"}
COLOR_MAPS = {"BASE_COLOR", *(name for name, spec in EXTRA_CHANNELS.items() if spec[2])}


def image_options(recipe, *, channel=None, hdr=False):
    """Use JPEG only for color; preserve data maps and HDR without loss."""
    fmt = "OPEN_EXR" if hdr else recipe.image_format
    if fmt == "JPEG" and channel not in COLOR_MAPS:
        fmt = "PNG"
    return {
        "format": fmt,
        "depth": (
            recipe.exr_depth
            if fmt == "OPEN_EXR"
            else "8"
            if fmt == "JPEG"
            else "16"
            if recipe.image_format == "JPEG"
            else recipe.image_depth
        ),
        "quality": recipe.jpeg_quality,
        "compression": recipe.png_compression,
        "exr_codec": recipe.exr_codec,
        "tiff_codec": recipe.tiff_codec,
    }
