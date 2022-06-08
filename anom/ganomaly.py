import logging
import warnings
from pathlib import Path


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings('ignore')
logger = logging.getLogger("anomalib")
from typing import Any, Dict

import numpy as np
from PIL import Image
from pytorch_lightning import Trainer
from torchvision.transforms import ToPILImage

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.pre_processing.transforms import Denormalize
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import get_experiment_logger


categories = (
    'bottle',
    'cable',
    'capsule',
    'carpet',
    'grid',
    'hazelnut',
    'leather',
    'metal_nut',
    'pill',
    'screw',
    'tile',
    'toothbrush',
    'transistor',
    'wood',
    'zipper',
)


def run():
    MODEL = 'ganomaly'  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
    CONFIG_PATH = f"/opt/project/anom/anomalib/anomalib/models/{MODEL}/config.yaml"
    config = get_configurable_parameters(config_path=CONFIG_PATH)
    config["dataset"]["path"] = "data"

    # assert category in categories
    # config["dataset"]["category"] = category
    datamodule = get_datamodule(config)
    datamodule.setup()
    datamodule.prepare_data()  # Create train/val/test/prediction sets.

    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    # start training
    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)

    load_model_callback = LoadModelCallback(
        weights_path=trainer.checkpoint_callback.best_model_path
    )
    trainer.callbacks.insert(0, load_model_callback)
    trainer.test(model=model, datamodule=datamodule)

    # print(config["project"]["path"])


if __name__ == '__main__':
    run()
