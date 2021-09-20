import torch
import torch.nn as nn
import torch.nn.functional as F

"""

    challenges:
        （1）previous approaches directly adopt deep architectures designed for category prediction to 
        pixel-wise labelling, results appear coarse because max pooling and sub-sampling reduce feature 
        map resolution.
        （2）hundreds of millions trainable parameters encounter difficulties in performing end-to-end 
        training.


    schemes:
        （1）adopt Encoder-Decoder model.
        （2）reusing max pooling indices in decoding process:
            1. improves boundary delineation,
            2. reduces the number of parameters enabling end-to-end training,
            3. incorporated into encoder-decoder architecture with a little modification.
        （3）widely used Fully Convolutional Network。
        （4）SGD-Momentum learning rate is a fixed 0.001，momentum is 0.9;mini batch is 12, training set
        is shuffled;select the cross entropy loss;adopts the median frequency balancing to balance class.


    architecture: 
        （1）Encoder network is identical to 13 Convolutional layers in VGG16;
        （2）Decoder network is to map the low resolution encoder feature map to full input resolution 
        feature maps for pixel-wise classification,the decoder uses pooling indices computed in the 
        max-pooling step of the corresponding encoder to perform non-linear upsampling;The upsampled 
        maps are sparse and convolved with trainable filters to produce dense feature maps.
        （3)no biases are used after the convolutions and no non-linearity in the decoder network.


    improvements:
        （1）Large number of classes in segmentation task performs worse.
        （2）smaller classes have lower accuracy.
        （3）the inability of deep network architecture.
        （4）SegNet can apply for real-time applications such as road scene understanding and AR.
        （5）End-to-end learning of deep segmentation architectures.


    results:
        （1）SegNet-EncoderAddition and FCN-Basic-NoDimReduction perform better than the other variants.
        （2）DenseCRF could be removed when sufficient amount of training data is made available.
        （3）SegNet could preform better when sufficient amount of training time is made available.

"""


class SegNet(nn.Module):

    _in_channels = 3

    def __init__(self, num_classes, norm_layer, dropout=0.5):
        super(SegNet, self).__init__()

        self.encoder1 = self._make_encoder(64, norm_layer, 2)
        self.encoder2 = self._make_encoder(128, norm_layer, 2)
        self.encoder3 = self._make_encoder(256, norm_layer, 3)
        self.encoder4 = self._make_encoder(512, norm_layer, 3)
        self.encoder5 = self._make_encoder(512, norm_layer, 3)

        self.decoder1 = self._make_decoder(512, norm_layer, 3, p=dropout)
        self.decoder2 = self._make_decoder(256, norm_layer, 3, p=dropout)
        self.decoder3 = self._make_decoder(128, norm_layer, 3, p=dropout)
        self.decoder4 = self._make_decoder(64, norm_layer, 2, p=dropout)
        self.decoder5 = self._make_decoder(num_classes, norm_layer, 2, p=dropout)


    def _make_encoder(self, channels, norm_layer, nums):
        layers = []
        for _ in range(nums):
            layers.append(nn.Conv2d(self._in_channels, channels, kernel_size=(3, 3),
                                     padding=1, bias=False))
            layers.append(norm_layer(channels))
            layers.append(nn.ReLU())
            self._in_channels = channels

        layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True))

        return nn.Sequential(*layers)


    def _make_decoder(self, channels, norm_layer, nums, p=0.5):
        layers = []
        for _ in range(nums):
            layers.append(nn.Conv2d(self._in_channels, channels, kernel_size=(3, 3),
                                     padding=1, bias=False))
            layers.append(norm_layer(channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=p))
            self._in_channels = channels

        return nn.Sequential(*layers)


    def forward(self, x):
        x, indices1 = self.encoder1(x)
        x, indices2 = self.encoder2(x)
        x, indices3 = self.encoder3(x)
        x, indices4 = self.encoder4(x)
        x, indices5 = self.encoder5(x)

        x = self.decoder1(F.max_unpool2d(x, kernel_size=(2, 2), indices=indices5))
        x = self.decoder2(F.max_unpool2d(x, kernel_size=(2, 2), indices=indices4))
        x = self.decoder3(F.max_unpool2d(x, kernel_size=(2, 2), indices=indices3))
        x = self.decoder4(F.max_unpool2d(x, kernel_size=(2, 2), indices=indices2))
        x = self.decoder5(F.max_unpool2d(x, kernel_size=(2, 2), indices=indices1))

        return x



if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = SegNet(num_classes=21, norm_layer=nn.BatchNorm2d)
    y = model(x)
    print(y.shape)



