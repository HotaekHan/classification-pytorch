# python
import os
import argparse
import random
import numpy as np
import shutil
from tqdm import tqdm

# pytorch
import torch
import torchvision.transforms as transforms
from torchvision import datasets


# 3rd-party utils
from torch.utils.tensorboard import SummaryWriter

# user-defined
from datagen import jsonDataset
import utils
import net_factory


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='path of config file')
opt = parser.parse_args()

config = utils.get_config(opt.config)

'''set random seed'''
random.seed(config['params']['seed'])
np.random.seed(config['params']['seed'])
torch.manual_seed(config['params']['seed'])
os.environ["PYTHONHASHSEED"] = str(config['params']['seed'])

'''cuda'''
if torch.cuda.is_available() and not config['gpu']['used']:
    print("WARNING: You have a CUDA device, so you should probably run with using cuda")

if isinstance(config['gpu']['ind'], list):
    cuda_str = 'cuda:' + str(config['gpu']['ind'][0])
elif isinstance(config['gpu']['ind'], int):
    cuda_str = 'cuda:' + str(config['gpu']['ind'])
else:
    raise ValueError('Check out gpu id in config')

device = torch.device(cuda_str if config['gpu']['used'] else "cpu")

'''Data'''
print('==> Preparing data..')
img_size = config['params']['image_size'].split('x')
img_size = (int(img_size[0]), int(img_size[1]))

if config['data']['name'] == 'cifar10' or config['data']['name'] == 'cifar100':
    transform_test = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_test = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.ToTensor()
    ])

def collate_fn_test(batch):
    imgs = [transform_test(x[0]) for x in batch]
    targets = [x[1] for x in batch]

    inputs = torch.stack(imgs)
    targets = torch.tensor(targets)

    return inputs, targets

if config['data']['name'] == 'cifar100':
    num_classes = 100
    train_data = datasets.CIFAR100(os.getcwd(), train=True, download=True, transform=None)
    num_train = len(train_data)
    num_valid = int(num_train * 0.2)
    num_train = num_train - num_valid

    train_dataset, valid_dataset = torch.utils.data.random_split(train_data, [num_train, num_valid])
    test_dataset = datasets.CIFAR100(os.getcwd(), train=False, download=True, transform=None)
elif config['data']['name'] == 'cifar10':
    num_classes = 10
    train_data = datasets.CIFAR10(os.getcwd(), train=True, download=True, transform=None)
    num_train = len(train_data)
    num_valid = int(num_train * 0.2)
    num_train = num_train - num_valid

    train_dataset, valid_dataset = torch.utils.data.random_split(train_data, [num_train, num_valid])
    test_dataset = datasets.CIFAR10(os.getcwd(), train=False, download=True, transform=None)
elif config['data']['name'] == 'its':
    target_classes = config['params']['classes'].split('|')
    num_classes = len(target_classes)
    train_dataset = jsonDataset(path=config['data']['train'].split(' ')[0], classes=target_classes,
                                transform=None,
                                input_image_size=img_size)
    valid_dataset = jsonDataset(path=config['data']['valid'].split(' ')[0], classes=target_classes,
                                transform=None,
                                input_image_size=img_size)
    test_dataset = jsonDataset(path=config['data']['test'].split(' ')[0], classes=target_classes,
                                transform=None,
                                input_image_size=img_size)
else:
    raise NotImplementedError('Unsupported Dataset: ' + str(config['data']['name']))

assert train_dataset
assert valid_dataset
assert test_dataset

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['params']['batch_size'],
    shuffle=False, num_workers=config['params']['workers'],
    collate_fn=collate_fn_test,
    pin_memory=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config['params']['batch_size'],
    shuffle=False, num_workers=config['params']['workers'],
    collate_fn=collate_fn_test,
    pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config['params']['batch_size'],
    shuffle=False, num_workers=config['params']['workers'],
    collate_fn=collate_fn_test,
    pin_memory=True)

dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

''' Model'''
net = net_factory.load_model(config=config, num_classes=num_classes)
net = net.to(device)
ckpt = torch.load(os.path.join(config['exp']['path'], 'best.pth'), map_location=device)
weights = utils._load_weights(ckpt['net'])
missing_keys = net.load_state_dict(weights, strict=False)
print(missing_keys)

'''print out net'''
num_parameters = 0.
for param in net.parameters():
    sizes = param.size()

    num_layer_param = 1.
    for size in sizes:
        num_layer_param *= size
    num_parameters += num_layer_param
print("num. of parameters : " + str(num_parameters))

'''print out'''
print("num. train data : " + str(len(train_dataset)))
print("num. valid data : " + str(len(valid_dataset)))
print("num. test data : " + str(len(test_dataset)))
print("num_classes : " + str(num_classes))

utils.print_config(config)

input("Press any key to continue..")

def view_inputs(x):
    import cv2

    x = x.detach().cpu().numpy()
    batch = x.shape[0]

    for iter_x in range(batch):
        np_x = x[iter_x]
        np_x = (np_x * 255.).astype(np.uint8)
        img = np.transpose(np_x, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('test', img)
        cv2.waitKey(0)


def do_test(phase):
    net.eval()
    phase_dataloader = dataloaders[phase]

    all_correct = 0
    all_samples = 0
    with torch.set_grad_enabled(False):
        # with autograd.detect_anomaly():
        for batch_idx, (inputs, targets) in enumerate(tqdm(phase_dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # view_inputs(inputs)
            logits = net(inputs)
            outputs = logits.log_softmax(dim=1)
            preds = outputs.argmax(dim=1, keepdim=False)

            all_correct += preds.eq(targets).float().sum()
            all_samples += inputs.shape[0]

    accuracy = all_correct / all_samples
    print('%-3s: %.3f' % (phase, accuracy))

if __name__ == '__main__':
    for dataset_name in dataloaders:
        print('Test on ' + str(dataset_name))
        do_test(dataset_name)