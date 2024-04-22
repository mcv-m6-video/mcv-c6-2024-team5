""" Functions to create models """

import torch
import torch.nn as nn
import torchvision

# from movinets import MoViNet
# from movinets.config import _C

def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    
    # ? Smaller or equal models

    elif model_name == 'movinet_a0':
        return create_movinet_a0(load_pretrain, num_classes)

    elif model_name == 'mobilenetV3_small':
        return create_mobilenetV3_small(load_pretrain, num_classes)
    
    # ? Temporal analysis

    elif model_name == 'efficientnet_b0':
        return create_efficientnet_b0(load_pretrain, num_classes)

    elif model_name == 'squeezenet':
        return create_squeezenet(load_pretrain, num_classes)

    elif model_name == 'shufflenet_v2':
        return create_shufflenet_v2(load_pretrain, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")

def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

# def create_movinet_a0(load_pretrain, num_classes):
#     # load movinet A0
#     model = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=load_pretrain)
#     model.classifier[3] = torch.nn.Conv3d(2048, num_classes, (1,1,1))
#     return model
#     # return nn.Sequential(
#     #     model,
#     #     nn.Linear(600, num_classes, bias=True),
#     # )

def create_mobilenetV3_small(load_pretrain, num_classes):
    model = torchvision.models.mobilenet_v3_small(pretrained=load_pretrain)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model

def create_efficientnet_b0(load_pretrain, num_classes):
    model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
    model._fc = nn.Linear(1280, num_classes)
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