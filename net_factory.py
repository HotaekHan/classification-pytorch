from models import ResNet, ResNetD, ShuffleNetV2


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
        pass
    elif config['model']['type'] == 'regnet':
        pass
    elif config['model']['type'] == 'resnest':
        pass
    elif config['model']['type'] == 'efficient':
        pass
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
    else:
        raise ValueError('Unsupported architecture: ' + str(config['model']['type']))

    return net

if __name__ == '__main__':
    net = load_model()