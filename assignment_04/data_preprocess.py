from pathlib import Path
from argparse import ArgumentParser
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

label = args.dir.name

images = [img for img in args.dir.iterdir()]

i = 1
while images:
    img_path = images.pop(0)
    new_name = img_path.parent.joinpath(f"{label}{i}{img_path.suffix}")
    if new_name.exists():
        images.remove(new_name)
        images.append(img_path)
    else:
        img_path.rename(new_name)
    apply_transforms(new_name)
    i += 1
