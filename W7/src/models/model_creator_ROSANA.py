""" Functions to create models """

import torch
import torch.nn as nn
import torchvision

from models.efficientnet_pytorch_3d.model import EfficientNet3D

# from efficientnet_pytorch_3d import EfficientNet3D
# from movinets import MoViNet
# from movinets.config import _C

def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    
    # ? Smaller or equal models

    # elif model_name == 'movinet_a0_stream':
    #     return create_movinet_a0_stream(load_pretrain, num_classes)

    elif model_name == 'mobilenetV3_small':
        return create_mobilenetV3_small(load_pretrain, num_classes)
    
    # ? Temporal analysis

    elif model_name == 'efficientnet_b0_3d':
        return create_efficientnet_b0_3d(load_pretrain, num_classes)

    elif model_name == 'squeezenet':
        return create_squeezenet(load_pretrain, num_classes)

    elif model_name == 'shufflenet_v2':
        return create_shufflenet_v2(load_pretrain, num_classes)
    
    elif model_name == 'resnet3d_18':
        return create_resnet3d_18(load_pretrain, num_classes)
    
    elif model_name == 'resnet_mc_18':
        return create_resnet_mc_18(load_pretrain, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")

def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

# def create_movinet_a0_stream(load_pretrain, num_classes):
#     # load movinet A0 Stream
#     model = MoViNet(_C.MODEL.MoViNetA0,pretrained = True)
#     return model

def create_mobilenetV3_small(load_pretrain, num_classes):
    model = torchvision.models.mobilenet_v3_small(pretrained=load_pretrain)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model

def create_efficientnet_b0_3d(load_pretrain, num_classes):
    # Load pretrained EfficientNet-B0 3D model
    model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': num_classes},in_channels=3)

    if load_pretrain:
        # Optionally load pretrained weights
        # Note: EfficientNet3D does not currently support pretrained weights
        pass

    return model

def create_squeezenet(load_pretrain, num_classes):
    model = torchvision.models.squeezenet1_1(pretrained=load_pretrain)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes
    return model

def create_shufflenet_v2(load_pretrain, num_classes):
    model = torchvision.models.shufflenet_v2_x1_5(pretrained=load_pretrain)
    model.fc = nn.Linear(1024, num_classes)
    return model

def create_resnet3d_18(load_pretrain, num_classes):
    model = torchvision.models.video.r3d_18(pretrained=load_pretrain)
    model.fc = nn.Linear(512, num_classes)
    return model

def create_resnet_mc_18(load_pretrain, num_classes):
    model = torchvision.models.video.mc3_18(pretrained=load_pretrain)
    model.fc = nn.Linear(512, num_classes)
    return model