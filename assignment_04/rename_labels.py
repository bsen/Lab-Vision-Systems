from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--dir", required=True, type=Path, help='Path of the images directory')
args = parser.parse_args()

if not args.dir.exists():
    raise FileNotFoundError(args.dir)

label = args.dir.name

images = [img for img in args.dir.iterdir()]

i = 1
while images:
    img = images.pop(0)
    new_name = img.parent.joinpath(f"{label}{i}{img.suffix}")
    if new_name.exists():
        images.remove(new_name)
        images.append(img)
    else:
        img.rename(new_name)
    i += 1
