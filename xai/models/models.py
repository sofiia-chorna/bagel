from xai.models.alexnet import AlexNet
from xai.models.resnet18 import ResNet18
from xai.models.vgg16 import VGG16
from xai.models.vit import ViT
from xai.utils.logger import logger


def get_model(model_name: str, num_classes: int):
    logger.info(f"Start loading model: {model_name}")

    match model_name:
        case "alexnet":
            return AlexNet(num_classes)

        case "resnet18":
            return ResNet18(num_classes)

        case "vgg16":
            return VGG16(num_classes)

        case "vit":
            return ViT(num_classes)

        case _:
            raise ValueError(f"'{model_name}' no such model available")

    logger.info(f"End loading model")
