import mitsuba as mi
import drjit as dr
from integrators.utils import mis_power_heuristic


class SRFEIntegrator(mi.SamplingIntegrator):
    """
    Stylized Rendering as Function Expectation (SRFE) integrator.

    This integrator is a proof-of-concept implementation based on the paper
    "Stylized Rendering as Function Expectation". It extends a basic path
    tracer to accumulate feature buffers that can be used for stylized
    rendering.
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        self.max_depth = props.get("max_depth", 8)  # Maximum depth for ray tracing
        self.rr_depth = props.get("rr_depth", 5)  # Russian Roulette depth

        # For simplicity, we will hardcode the stylization function for now.
        # A more advanced implementation would allow loading these from the scene file.
        self.stylization_func = self.simple_stylization

    @dr.syntax
    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        medium: mi.Medium = None,
        active: mi.Mask = True,
    ) -> tuple[mi.Color3f, mi.Mask, list[mi.Color3f]]:
        """
        Samples the incident radiance along a ray and accumulates feature buffers.
        """

        # Path tracer internal state
        ray = mi.Ray3f(ray)
        throughput = mi.Spectrum(1.0)
        L = mi.Spectrum(0.0)
        depth = mi.UInt32(0)
        eta = mi.Float(1.0)
        active = mi.Mask(active)

        bsdf_context = mi.BSDFContext()

        # Previous hit information for MIS
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        while active:
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()

            # Previous hit direct illumination
            # BSDF MIS
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            mis_weight_bsdf = mis_power_heuristic(
                1.0,
                prev_bsdf_pdf,
                1.0,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta),
            )
            Le = throughput * mis_weight_bsdf * ds.emitter.eval(si)

            # Emitter sampling
            bsdf = si.bsdf(ray)
            # Only sample if the BSDF is not delta
            active_es = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            ds, sample_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_es
            )
            wo = si.to_local(ds.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_context, si, wo, active_es)
            mis_weight_light = dr.select(
                ds.delta, 1.0, mis_power_heuristic(1.0, ds.pdf, 1.0, bsdf_pdf)
            )
            Lr_dir = throughput * bsdf_val * sample_weight * mis_weight_light

            # BSDF sampling
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_context,
                si,
                sampler.next_1d(active),
                sampler.next_2d(active),
                active,
            )
            active &= bsdf_sample.pdf > 0.0

            # Update state
            L += Le + Lr_dir
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            eta *= bsdf_sample.eta
            throughput *= bsdf_weight

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # Russian Roulette
            if depth >= self.rr_depth:
                # Apply Russian Roulette to terminate paths
                rr_prob = dr.minimum(dr.max(throughput), 0.95)
                active &= sampler.next_1d(active) < rr_prob
                throughput *= dr.rcp(rr_prob)

            depth += 1
            if depth >= self.max_depth:
                active &= False

        # Apply stylization function
        stylized_result = self.stylization_func(L)

        return stylized_result, depth != 0, []

    def simple_stylization(self, radiance):
        """
        A simple example of a stylization function.
        This function just returns the computed radiance.
        A more complex function could use the feature buffers to modify the output.
        """
        return radiance


# Register the integrator with Mitsuba
mi.register_integrator("srfe", lambda props: SRFEIntegrator(props))
