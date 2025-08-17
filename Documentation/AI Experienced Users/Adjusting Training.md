# Adjusting Training Schedules in C3LLS

For users with experience in AI, the C3LLS pipeline is modifiable. 

The backbone of C3LLS (pre-processing, training, and predictions) are done within [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)

This page reviews common questions that might come up with adjusting model architecture/hyper-parameters.

## How to adjust model hyperparameters?

Training within nnUNetV2 is built around a base training [class](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py). This defines strategies for 

* Data augmentation
* Loss functions
* LR schedulers
* Network architectures
* Optimizer
* Training Length
* A list of hyper-parameters found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L144)

The most straightforward way to adjust hyper-parameters is through class inheritance. Examples can be found [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py) for adjusting training length.

Likewise, you can define your own data augmentation strategies, loss functions, LR schedulers, etc. following the example [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/lr_schedule/nnUNetTrainerCosAnneal.py)


