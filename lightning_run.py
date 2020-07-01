import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import models.ResNet as ResNet

class ClassificationPL(pl.LightningModule):
    def __init__(self, net, configs):
        super(ClassificationPL, self).__init__()
        self.model = net
        self.configs = configs

        if self.configs['params']['loss'] == 'CE':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            raise ValueError('Unsupported loss: ' + str(self.configs['params']['loss']))

    def prepare_data(self) -> None:
        # In DDP, this function is called once in total at GLOBAL_RANK=0
        # If, prepare_data_per_node is True, called once per node at LOCAL_RANK=0
        # # good
        # download_data()
        # tokenize()
        # etc()
        #
        # # bad
        # self.split = data_split
        # self.some_state = some_other_state()

        # download dataset
        if self.configs['data']['name'] == 'cifar100':
            datasets.CIFAR100(os.getcwd(), train=True, download=True)
            datasets.CIFAR100(os.getcwd(), train=False, download=True)
        else:
            raise NotImplementedError('Not supported dataset:' + str(self.configs['data']['name']))

    def setup(self, stage: str):
        # Example
        # # step is either 'fit' or 'test' 90% of the time not relevant
        # data = load_data()
        # num_classes = data.classes
        # self.l1 = nn.Linear(..., num_classes)

        print('call setup: ' + str(stage))

        if stage == 'fit':
            if self.configs['data']['name'] == 'cifar100':
                self.train_data = datasets.CIFAR100(os.getcwd(), train=True, download=False,
                                                    transform=transforms.ToTensor())
                num_train = len(self.train_data)
                num_valid = int(num_train * 0.1)
                num_train = num_train - num_valid

                self.train_data, self.valid_data = torch.utils.data.random_split(self.train_data,
                                                                                 [num_train, num_valid])
            else:
                raise NotImplementedError('Not supported dataset:' + str(self.configs['data']['name']))
        elif stage == 'test':
            if self.configs['data']['name'] == 'cifar100':
                self.test_data = datasets.CIFAR100(os.getcwd(), train=False, download=False,
                                                   transform=transforms.ToTensor())
            else:
                raise NotImplementedError('Not supported dataset:' + str(self.configs['data']['name']))
        else:
            raise ValueError('Unexpected stage: ' + str(stage))

    def configure_optimizers(self):
        if self.configs['optimizer']['type'] == 'Adam':
            return optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=float(self.configs['optimizer']['lr']))
        elif self.configs['optimizer']['type'] == 'SGD':
            return optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                             lr=float(self.configs['optimizer']['lr']),
                             momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=int(self.configs['params']['batch_size']),
            shuffle=True,
            num_workers=int(self.configs['params']['workers']),
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=int(self.configs['params']['batch_size']),
            shuffle=False,
            num_workers=int(self.configs['params']['workers']),
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=int(self.configs['params']['batch_size']),
            shuffle=False,
            num_workers=int(self.configs['params']['workers']),
            pin_memory=True
        )

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)
        logs = {'training_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)
        logs = {'valid_loss': loss}
        # pred = output.argmax(dim=1, keepdim=True)
        # accuracy = pred.eq(target.view_as(pred)).float().mean()
        return {'val_loss': loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        # accuracy = sum(x['batch_val_acc'] for x in outputs) / len(outputs)
        loss = sum(x['val_loss'] for x in outputs) / len(outputs)
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        return {'avg_val_loss': loss}

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        return {'batch_test_acc': accuracy}

    def test_epoch_end(self, outputs):
        accuracy = sum(x['batch_test_acc'] for x in outputs) / len(outputs)
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        return {'log': {'test_acc': accuracy}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path of config file')
    opt = parser.parse_args()

    pl.seed_everything(1020)

    config = utils.get_config(opt.config)

    gpu_id = None
    if config['gpu']['used'] is True and torch.cuda.is_available():
        gpu_id = int(config['gpu']['ind'])

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(config['exp']['path'], '\\'),
                                          monitor='avg_val_loss',
                                          verbose=False,
                                          save_last=True,
                                          save_top_k=1,
                                          mode='min',
                                          save_weights_only=False,
                                          period=1)

    logger = TensorBoardLogger(save_dir=os.path.dirname(config['exp']['path']),
                               name=os.path.basename(config['exp']['path']),
                               version='log')

    net = ResNet.resnet18(pretrained=True, progress=True, num_classes=100)
    model = ClassificationPL(net=net, configs=config)

    trainer = pl.Trainer(fast_dev_run=False,
                         max_epochs=2,
                         precision=32,
                         check_val_every_n_epoch=1,
                         distributed_backend=None,
                         benchmark=True,
                         gpus=gpu_id,
                         limit_test_batches=1.0,
                         limit_val_batches=1.0,
                         log_save_interval=1,
                         row_log_interval=1,
                         logger=logger,
                         checkpoint_callback=checkpoint_callback
                         )

    trainer.fit(model=model)
    trainer.test(model=model)
