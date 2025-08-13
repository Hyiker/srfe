import mitsuba as mi
import drjit as dr
from mitsuba import Float, Vector3f, Thread, Vector1f, Ray3f
from mitsuba import SurfaceInteraction3f, PositionSample3f
from mitsuba import (
    BSDF,
    BSDFContext,
    BSDFFlags,
    has_flag,
    register_integrator,
)


def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    return dr.select(pdf_a > 0.0, pdf_a / (pdf_a + pdf_b), Float(0.0))


class SRFEIntegrator(mi.SamplingIntegrator):
    """
    Stylized Rendering as Function Expectation (SRFE) integrator.
    """

    def __init__(self, props):
        super().__init__(props)

    def sample(self, scene, sampler, rays, medium=None, active=True):
        return mi.Color3d(0.0), False, []


# Register the integrator with Mitsuba
mi.register_integrator("srfe", lambda props: SRFEIntegrator(props))
