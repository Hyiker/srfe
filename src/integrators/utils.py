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
