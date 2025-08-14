import mitsuba as mi
from tqdm import tqdm
import os

mi.set_variant("llvm_ad_rgb")

import drjit as dr
from argparse import ArgumentParser
from integrators import SRFEIntegrator


def main():
    parser = ArgumentParser(description="Render a scene using Mitsuba")
    parser.add_argument("scene", help="Path to the scene XML file")
    parser.add_argument(
        "--outdir", type=str, default=".", help="Output directory for rendered images"
    )
    parser.add_argument("--spp", type=int, default=128, help="Samples per pixel")
    args = parser.parse_args()

    # Load the scene XML and override the integrator
    scene = mi.load_file(args.scene)

    # Render the scene with tqdm progress
    spp = args.spp  # samples per pixel, adjust as needed
    image = None
    with tqdm(total=spp, desc="Rendering", unit="spp") as pbar:
        for i in range(spp):
            if image is None:
                image = mi.render(scene, spp=1, seed=i, sensor=0)
            else:
                image += mi.render(scene, spp=1, seed=i, sensor=0)
            pbar.update(1)
    image /= spp
    mi.util.write_bitmap(os.path.join(args.outdir, "output.exr"), image)
    print(f"Rendered image saved to {os.path.join(args.outdir, 'output.exr')}")


if __name__ == "__main__":
    main()
