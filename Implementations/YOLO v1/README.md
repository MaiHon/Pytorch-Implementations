# _*YOLO version 1 Implementation*_

Implementing YOLO version 1 Networks in [**pytorch**](https://pytorch.org).  
Welcome any advice with widely open arms.
<br></br>


- You Only Look Once: Unified, Real-Time Object Detection
- Authors
  - [Joseph Redomn | Santosh Divvala | Ross Girshick | Ali Farhadi]
  <br></br>
- [[**Paper**]](https://arxiv.org/abs/1506.02640)


<br></br>
# Todo
- [x] Dataset arange
- [x] Yolo on Pretrained VGG16-BN
- [ ] Yolo Training...
- [ ] Train Darkent on ImageNet Dataset 
- [ ] Yolo on Darknet training

<br></br>
# Training Conditions

- Dataset
  - PASCAL 2012 including validation set 

- Feature map extractor
  - VGG16 features
    - I didn't have IMAGENET datset for training Darknet...
  - 13 layers are freezed(hardware limitation)

- Batch size
  - 24(hardware limitation)

- Optimizer
  - **Same as original paper**
  - SGD
  - weight decay = 0.0005
  - momentum = 0.9
  - Learning Rate
    - Initialize 0.0001
    - First Epoch - slowly raise from 0.0001 to 0.001
    - Up to 75 epoch 0.001
    - for the another 30 epoch set to 0.0001
    - for the last 30 epoch set to 0.00001