import mitsuba as mi
import drjit as dr
from integrators.utils import mis_power_heuristic


class SRFEIntegrator(mi.SamplingIntegrator):
    """
    Stylized Rendering as Function Expectation (SRFE) integrator.

    This integrator is a proof-of-concept implementation based on the paper
    "Stylized Rendering as Function Expectation". It extends a basic path
    tracer to accumulate feature buffers that can be used for stylized
    rendering. This version is implemented recursively.
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)

        # ------ Path tracer params ------
        self.max_depth = props.get("max_depth", 8)
        self.rr_depth = props.get("rr_depth", 5)

        # ------ Stylization params ------
        self.stylize_depth = props.get("stylize_depth", 3)  # Maximum stylization depth
        self.ge_sample_count = props.get(
            "ge_sample_count", 4
        )  # Stylization group estimator samples

    def grayscale_stylize(self, radiance: mi.Spectrum) -> mi.Spectrum:
        color = mi.Color3f(radiance)
        # Apply a simple grayscale gradient stylization
        gray = 0.2989 * color.x + 0.5870 * color.y + 0.1140 * color.z
        return mi.Spectrum(gray)

    @dr.syntax(recursive=True)
    def style_shading(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        depth: mi.UInt32,
        eta: mi.Float,
        active: mi.Mask,
        prev_si: mi.SurfaceInteraction3f,
        prev_bsdf_pdf: mi.Float,
        prev_bsdf_delta: mi.Bool,
    ) -> tuple[mi.Spectrum, mi.Mask]:

        # Hit max depth
        if dr.hint(dr.none(active) or depth >= self.max_depth, mode="scalar"):
            return mi.Spectrum(0.0), mi.Mask(False)

        si = scene.ray_intersect(ray, active)
        active &= si.is_valid()

        # BSDF MIS
        ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
        mis_weight_bsdf = mis_power_heuristic(
            1.0,
            prev_bsdf_pdf,
            1.0,
            scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta),
        )

        # Russian Roulette
        rr_prob = dr.select(depth >= self.rr_depth, 0.85, 1.0)
        pass_rr = sampler.next_1d(active) < rr_prob
        if dr.hint(dr.none(pass_rr), mode="scalar"):
            return mi.Spectrum(0.0), mi.Mask(False)

        bsdf_context = mi.BSDFContext()
        L_dir_base = dr.select(active, mis_weight_bsdf * ds.emitter.eval(si), 0.0)
        L = mi.Spectrum(0.0)

        stylize_sample = dr.select(depth < self.stylize_depth, self.ge_sample_count, 1)
        sample_idx = dr.copy(stylize_sample)

        while dr.hint(sample_idx > 0, mode="evaluated"):
            # Direct illumination contribution
            L_dir = dr.detach(L_dir_base)

            # Emitter sampling
            bsdf = si.bsdf(ray)
            active_es = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            ds, sample_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_es
            )
            wo = si.to_local(ds.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_context, si, wo, active_es)
            mis_weight_light = dr.select(
                ds.delta, 1.0, mis_power_heuristic(1.0, ds.pdf, 1.0, bsdf_pdf)
            )
            L_dir += dr.select(
                active_es, bsdf_val * sample_weight * mis_weight_light, 0.0
            )

            # BSDF sampling for recursive step
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_context,
                si,
                sampler.next_1d(active),
                sampler.next_2d(active),
                active,
            )

            # Spawn new ray for recursion
            ray_next = si.spawn_ray(si.to_world(bsdf_sample.wo))

            # Recursive call
            L_indirect, _ = self.style_shading(
                scene,
                sampler,
                ray_next,
                depth + 1,
                eta * bsdf_sample.eta,
                active & (bsdf_sample.pdf > 0.0),
                dr.detach(si, True),
                bsdf_sample.pdf,
                mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta),
            )

            # Total outgoing
            L_sample = L_dir + bsdf_weight * L_indirect * dr.rcp(rr_prob)
            L_sample = self.grayscale_stylize(L_sample)
            L += L_sample
            sample_idx -= 1

        L /= mi.Float32(stylize_sample)

        return L, active

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
        Initializes the path tracing state and starts the recursive process.
        """

        L, valid = self.style_shading(
            scene=scene,
            sampler=sampler,
            ray=mi.Ray3f(ray),
            depth=mi.UInt32(0),
            eta=mi.Float(1.0),
            active=mi.Mask(active),
            prev_si=dr.zeros(mi.SurfaceInteraction3f),
            prev_bsdf_pdf=mi.Float(1.0),
            prev_bsdf_delta=mi.Bool(True),
        )

        return L, valid, []


# Register the integrator with Mitsuba
mi.register_integrator("srfe", lambda props: SRFEIntegrator(props))
