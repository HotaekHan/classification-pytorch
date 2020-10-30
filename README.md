## Classification playground

### Dataset
* Cifar 100

### Hyperparameter
* Optimizer  
  - SGD, lr=0.4, momentum=0.9, weight_decay=1e-4
  - Warmup_epoch=5, warmup_factor=0.1
  - Cosine LR decay
* Hyperparameters  
  - Batch size: 128
  - Input size: 32x32
  - Epoch: 300
* Regularizer  
  - Augmentation: RandomCrop, HorizontalFlip
  - Mixup alpha: 0.2
  - Label smoothing: 0.1
  - Early stopping
* Scratch Learning(without ImageNet pretrained weights)

Experiments
-----------------------------------
### Cifar 100
|Model|Accuracy(train)|Accuracy(test)|Num. of params|Elapsed time|Elapsed time(CPU)|
|------------|-----|-----|-----|-----|-----|
|ResNet50|0.99972|0.6136|23.7M|27.3321|138.6245|
|ResNeXt50|0.99972|0.6276|23.1M|27.6215|153.159|
|ResNet50D|0.99972|**0.6906**|23.7M|27.8048|148.4096|
|TResNet|0.9997|0.6064|29.5M|35.8719|164.0637|
|EfficientNet-B3|0.83825|0.6359|**10.8M**|27.5409|144.9197|
|EfficientNet-B4|0.71955|0.5674|17.7M|41.8908|187.9967|
|RegNetX-4.0gf|0.99972|0.6233|20.8M|**27.0687**|**104.3127**|
|RegNetX-6.4gf|0.99972|0.6757|24.7M|27.47|156.1908|
|RegNetY-4.0gf|0.99972|0.6072|19.6M|33.8534|166.9389|
|RegNetY-6.4gf|0.99972|0.6222|29.4M|47.6946|241.303|
|ResNeSt50|0.99972|0.6559|25.6M|40.8935|191.4831|

Table 1. Result of normal model

|Model|Accuracy(train)|Accuracy(test)|Num. of params|Elapsed time|Elapsed time(CPU)|
|------------|-----|-----|-----|-----|-----|
|ShuffleNetV2_2.0|0.9995|0.5122|5.5M|13.1761|54.6879|
|MobileNetV3_Small_1.0|0.73798|0.4875|**1.6M**|10.6767|46.5825|
|MobileNetV3_Large_1.0|0.79205|0.5092|4.3M|**10.2918**|62.6109|
|EfficientNet-B0|0.8143|0.5961|4.1M|13.348|76.6069|
|RegNetX-200mf|0.9057|0.5406|2.3M|17.4331|**42.2517**|
|RegNetY-200mf|0.8809|0.5335|2.8M|21.051|61.5444|
|ReXNet1.0x|0.9984|**0.6315**|3.6M|13.7785|95.1711|

Table 2. Result of light model