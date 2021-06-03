from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--dir", required=True, type=Path, help="Path of the images directory"
)
args = parser.parse_args()

if not args.dir.exists():
    raise FileNotFoundError(args.dir)

label = args.dir.name

for i, img_path in enumerate(args.dir.iterdir()):
    img_path.rename(img_path.parent.joinpath(f"{label}{i+1}{img_path.suffix}"))
