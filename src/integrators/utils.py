import mitsuba as mi
import drjit as dr


def mis_power_heuristic(n1, pdf1, n2, pdf2):
    """
    Apply the Multiple Importance Sampling (MIS) power heuristic.
    """
    # Compute the MIS weight
    w1 = dr.sqr(n1 * pdf1)
    pdf2 = n2 * pdf2

    return dr.detach(dr.select(pdf1 > 0, w1 / dr.fma(pdf2, pdf2, w1), 0), True)


def load_texture2d(path):
    bitmap = mi.Bitmap(path)
    bitmap = bitmap.convert(
        mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, srgb_gamma=True
    )
    tensor = mi.TensorXf(bitmap)
    return mi.Texture2f(tensor)
