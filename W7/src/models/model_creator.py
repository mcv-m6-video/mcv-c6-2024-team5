""" Functions to create models """

import torch
import torch.nn as nn
import torchvision

from movinets import MoViNet
from movinets.config import _C


class MM_model(nn.Module):
    def __init__(self, model_rgb, model_of, num_classes):
        super(MM_model, self).__init__()
        self.model_rgb = model_rgb
        self.model_of = model_of
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x_rgb, x_of):
        x_rgb = self.model_rgb(x_rgb)
        x_of = self.model_of(x_of)
        x = torch.cat((x_rgb, x_of), dim=1)
        x = self.fc(x)
        return x

    def freeze(self, num_layers):
        """
        Function to freeze the first num_layers layers of both the RGB and OF models (fc layer will always be unfrozen)
        :param num_layers: Number of layers to freeze in percentage
        :return:
        """
        layers_to_freeze = int(num_layers * len(list(self.model_rgb.parameters())))
        for i, param in enumerate(self.model_rgb.parameters()):
            if i < layers_to_freeze:
                param.requires_grad = False
        for i, param in enumerate(self.model_of.parameters()):
            if i < layers_to_freeze:
                param.requires_grad = False


def create_mm(path_rgb, path_of, num_classes):
    model_rgb = create_x3ds_weights(num_classes, path_rgb)
    model_of = create_x3ds_weights(num_classes, path_of)

    # Get rid of the last layer of the models
    model_rgb.fc = nn.Identity()
    model_of.fc = nn.Identity()

    final_model = MM_model(model_rgb, model_of, num_classes)
    return final_model


def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    
    # ? Smaller or equal models

    elif model_name == 'movinet_a0':
        return create_movinet_a0(load_pretrain, num_classes)

    elif model_name == 'mobilenetV3_small':
        return create_mobilenetV3_small(load_pretrain, num_classes)
    
    # ? No limit models

    elif model_name == 'x3d_s':
        return create_x3d_s(load_pretrain, num_classes)

    elif model_name == 'x3d_m':
        return create_x3d_m(load_pretrain, num_classes)

    elif model_name == 'resnet3d_18':
        return create_resnet3d_18(load_pretrain, num_classes)

    elif model_name == 'resnet_mc_18':
        return create_resnet_mc_18(load_pretrain, num_classes)
    
    # ? Temporal analysis

    elif model_name == 'efficientnet_b0':
        return create_efficientnet_b0(load_pretrain, num_classes)

    elif model_name == 'squeezenet':
        return create_squeezenet(load_pretrain, num_classes)

    elif model_name == 'shufflenet_v2':
        return create_shufflenet_v2(load_pretrain, num_classes)

    elif model_name == 'resnet50':
        return create_resnet50(load_pretrain, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")

def create_x3d_s(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

def create_x3d_m(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )


def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

def create_movinet_a0(load_pretrain, num_classes):
    # load movinet A0
    model = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=load_pretrain)
    model.classifier[3] = torch.nn.Conv3d(2048, num_classes, (1,1,1))
    return model
    # return nn.Sequential(
    #     model,
    #     nn.Linear(600, num_classes, bias=True),
    # )

def create_mobilenetV3_small(load_pretrain, num_classes):
    model = torchvision.models.mobilenet_v3_small(pretrained=load_pretrain)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model

def create_resnet3d_18(load_pretrain, num_classes):
    model = torchvision.models.video.r3d_18(pretrained=load_pretrain)
    model.fc = nn.Linear(512, num_classes)
    return model

def create_resnet_mc_18(load_pretrain, num_classes):
    model = torchvision.models.video.mc3_18(pretrained=load_pretrain)
    model.fc = nn.Linear(512, num_classes)
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

def create_resnet50(load_pretrain, num_classes):
    model = torchvision.models.resnet50(pretrained=load_pretrain)
    model.fc = nn.Linear(2048, num_classes)
    return model

def create_x3ds_weights(num_classes, path):
    model = create_x3d_s(False, num_classes)
    model.load_state_dict(torch.load(path))
    return model