import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

import drjit as dr
from argparse import ArgumentParser
from integrators import SRFEIntegrator


def create_integrator():
    return mi.load_dict(
        {
            "type": "srfe",
            # Add custom parameters here if needed
        }
    )


def main():
    parser = ArgumentParser(description="Render a scene using Mitsuba")
    parser.add_argument("scene", help="Path to the scene XML file")
    args = parser.parse_args()

    # Load the scene XML and override the integrator
    scene = mi.load_file(args.scene)

    # Render the scene
    image = mi.render(scene=scene, integrator=create_integrator())
    mi.util.write_bitmap("output.exr", image)


if __name__ == "__main__":
    main()
