from models import ResNet, ResNetD, ShuffleNetV2, TResNet, Mobilenetv3, EfficientNet, RegNet


def load_model(config, num_classes):
    if config['model']['type'] == 'resnet':
        if config['model']['arch'] == 'resnet50':
            net = ResNet.resnet50(pretrained=False, progress=False, num_classes=num_classes)
        elif config['model']['arch'] == 'resnext50':
            net = ResNet.resnext50_32x4d(pretrained=False, progress=False, num_classes=num_classes)
        elif config['model']['arch'] == 'resnet50d':
            net = ResNetD.resnet50d(pretrained=False, progress=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'tresnet':
        if config['model']['arch'] == 'tresnetm':
            net = TResNet.TResnetM(num_classes=num_classes)
        elif config['model']['arch'] == 'tresnetl':
            net = TResNet.TResnetL(num_classes=num_classes)
        elif config['model']['arch'] == 'tresnetxl':
            net = TResNet.TResnetXL(num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'regnet':
        if config['model']['arch'] == 'regnetx':
            net = RegNet.RegNet(num_classes=num_classes, RegNetY=False)
        elif config['model']['arch'] == 'regnety':
            net = RegNet.RegNet(num_classes=num_classes, RegNetY=True)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'resnest':
        pass
    elif config['model']['type'] == 'efficient':
        if config['model']['arch'] == 'b0':
            net = EfficientNet.efficientnet_b0(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b1':
            net = EfficientNet.efficientnet_b1(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b2':
            net = EfficientNet.efficientnet_b2(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b3':
            net = EfficientNet.efficientnet_b3(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b4':
            net = EfficientNet.efficientnet_b4(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b5':
            net = EfficientNet.efficientnet_b5(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'b6':
            net = EfficientNet.efficientnet_b6(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'assembled':
        pass
    elif config['model']['type'] == 'shufflenet':
        if config['model']['arch'] == 'v2_x0_5':
            net = ShuffleNetV2.shufflenet_v2_x0_5(pretrained=False, progress=False, num_classes=num_classes)
        elif config['model']['arch'] == 'v2_x1_0':
            net = ShuffleNetV2.shufflenet_v2_x1_0(pretrained=False, progress=False, num_classes=num_classes)
        elif config['model']['arch'] == 'v2_x1_5':
            net = ShuffleNetV2.shufflenet_v2_x1_5(pretrained=False, progress=False, num_classes=num_classes)
        elif config['model']['arch'] == 'v2_x2_0':
            net = ShuffleNetV2.shufflenet_v2_x2_0(pretrained=False, progress=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    elif config['model']['type'] == 'mobilenet':
        if config['model']['arch'] == 'small_075':
            net = Mobilenetv3.mobilenetv3_small_075(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'small_100':
            net = Mobilenetv3.mobilenetv3_small_100(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'large_075':
            net = Mobilenetv3.mobilenetv3_large_075(pretrained=False, num_classes=num_classes)
        elif config['model']['arch'] == 'large_100':
            net = Mobilenetv3.mobilenetv3_large_100(pretrained=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['model']['arch']))
    else:
        raise ValueError('Unsupported architecture: ' + str(config['model']['type']))

    return net

if __name__ == '__main__':
    import torch

    config = {'model':
                  {'type':'regnet',
                   'arch':'regnety'}}

    net = load_model(config, 100)

    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param

    print(net)
    print("num. of parameters : " + str(num_parameters))

    out = net((torch.randn(10, 3, 32, 32)))
    print(out.size())