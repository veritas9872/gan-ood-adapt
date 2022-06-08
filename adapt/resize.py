from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


def downsize():
    source = '/opt/project/data/transistor/train/good'
    target = '/opt/project/adapt/images/transistor/train/good'
    Path(target).mkdir(parents=True, exist_ok=True)

    for file in tqdm(Path(source).glob('*.png')):
        img = Image.open(str(file))
        img = img.resize(size=(32, 32), resample=Image.LANCZOS, reducing_gap=3)
        save_path = Path(target, file.name)
        img.save(save_path)


def upsize():
    source = '/opt/project/adapt/output/transistor/train/good'
    target = '/opt/project/adapt/recons/transistor/train/good'
    Path(target).mkdir(parents=True, exist_ok=True)

    for file in tqdm(Path(source).glob('*.png')):
        img = Image.open(str(file))
        img = img.resize(size=(1024, 1024), resample=Image.LANCZOS, reducing_gap=3)
        save_path = Path(target, file.name)
        img.save(save_path)


if __name__ == '__main__':
    # downsize()
    upsize()
