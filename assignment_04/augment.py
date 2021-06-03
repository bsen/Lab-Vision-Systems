from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

transforms = [
    Image.FLIP_TOP_BOTTOM,
    Image.FLIP_LEFT_RIGHT,
    Image.ROTATE_90,
    Image.ROTATE_180,
    Image.ROTATE_270,
]


def apply_transforms(img_path: Path):
    img = Image.open(img_path)
    for i, t in enumerate(transforms):
        aug_path = img_path.parent.joinpath(
            f"{img_path.stem}_aug{i+1}{img_path.suffix}"
        )
        img.transpose(method=t).save(aug_path)


parser = ArgumentParser()
parser.add_argument(
    "--dir", required=True, type=Path, help="Path of the images directory"
)
args = parser.parse_args()

if not args.dir.exists():
    raise FileNotFoundError(args.dir)

for img_path in args.dir.iterdir():
    apply_transforms(img_path)
