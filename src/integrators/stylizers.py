import mitsuba as mi
import drjit as dr
import integrators.utils as utils


class Stylizer:
    def __init__(self):
        pass

    def apply(self, radiance: mi.Spectrum) -> mi.Spectrum:
        raise NotImplementedError


class GrayscaleStylizer(Stylizer):
    def __init__(self, props: mi.Properties):
        super().__init__()

    def apply(self, radiance: mi.Spectrum) -> mi.Spectrum:
        color = mi.Color3f(radiance)
        gray = mi.srgb_to_xyz(color).y
        return mi.Spectrum(gray)


class ACPStylizer(Stylizer):
    """
    Color Remapping that Affects the Color on a Path Trace(ACP) stylizer.
    From Doi'21 "Global Illumination-Aware Stylised Shading."
    """

    def __init__(self, props: mi.Properties):
        super().__init__()
        self.colormap = utils.load_texture2d(props.get("acp_colormap"))
        self.w_min = props.get("acp_w_min", 0.01)  # Minimum weight
        self.y_min = props.get(
            "acp_y_min", 0.0
        )  # Minimum y value(for texture sampling)
        self.y_max = props.get("acp_y_max", 1.0)  # Maximum y value

    def sample_colormap(self, u: float) -> mi.Color3f:
        width = self.colormap.shape[0]
        return mi.Color3f(self.colormap.eval(mi.Vector2f(u * width, 0.0)))

    def apply(self, radiance: mi.Spectrum) -> mi.Spectrum:
        color = mi.Color3f(radiance)
        y = mi.srgb_to_xyz(color).y
        w = dr.maximum(y, self.w_min)
        u = dr.clamp((y - self.y_min) / (self.y_max - self.y_min), 0.0, 1.0)
        cm_color = self.sample_colormap(u)
        # Apply the ACP stylization using the colormap
        return mi.Spectrum(cm_color * w)


def create_stylizer(stylizer_type: str, props: mi.Properties) -> Stylizer:
    if stylizer_type == "grayscale":
        return GrayscaleStylizer(props)
    elif stylizer_type == "ACP":
        return ACPStylizer(props)
    raise ValueError(f"Unknown stylizer type: {stylizer_type}")
