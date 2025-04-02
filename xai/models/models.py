from typing import Optional

from xai.models.alexnet import AlexNet
from xai.models.base_model import BaseModel
from xai.models.densenet import DenseNet
from xai.models.efficientnet import EfficientNet
from xai.models.googlelenet import GoogLeNet
from xai.models.inceptionv3 import InceptionV3
from xai.models.resnet18 import ResNet18
from xai.models.resnext import ResNeXt
from xai.models.vgg16 import VGG16
from xai.models.vit import ViT
from xai.utils.logger import logger


def get_model(
    model_name: str, num_classes: int, checkpoint: Optional[str] = None
) -> BaseModel:
    logger.info(f"Start loading model: {model_name}")

    match model_name:
        case "alexnet":
            return AlexNet(num_classes, checkpoint)

        case "resnet18":
            return ResNet18(num_classes, checkpoint)

        case "vgg16":
            return VGG16(num_classes, checkpoint)

        case "vit":
            return ViT(num_classes, checkpoint)

        case "densenet121":
            return DenseNet(num_classes, type="densenet121", checkpoint_path=checkpoint)

        case "densenet169":
            return DenseNet(num_classes, type="densenet169", checkpoint_path=checkpoint)

        case "efficientnet":
            return EfficientNet(num_classes, checkpoint)

        case "inceptionv3":
            return InceptionV3(num_classes, checkpoint)

        case "googlelenet":
            return GoogLeNet(num_classes, checkpoint)

        case "resnext50":
            return ResNeXt(num_classes, type="resnext50", checkpoint_path=checkpoint)

        case "resnext101":
            return ResNeXt(num_classes, type="resnext101", checkpoint_path=checkpoint)

        case _:
            raise ValueError(f"'{model_name}' no such model available")

    logger.info(f"End loading model")
