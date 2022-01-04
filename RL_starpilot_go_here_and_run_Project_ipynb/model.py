from libs.joon.misc_util import xavier_uniform_init
import torch.nn as nn
import torch

#halfs the dim size
class My_NN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(My_NN_Block, self).__init__()
        #bring the number of channel to out_channel
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        #keep the number of channels for these conv2d's
        #kernal 3, stride 1 and padding 1 also keeps the dimensions the sames
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,  kernel_size=3, stride=1, padding=1)
        #kernal 6, stride 2 and padding 2 halfs the size
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,  kernel_size=6, stride=2, padding=2)
        #kernal 3, stride 1 and padding 1 keeps the dimensions the sames
        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,  kernel_size=3, stride=1, padding=1)
        #kernal 3, stride 1 and padding 1 keeps the dimensions the sames
        self.conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        
        x = self.conv1(x);
        
        x = nn.ReLU()(x);
        x = self.conv2(x);
        x = nn.ReLU()(x);
        x = self.conv3(x);

        x = nn.ReLU()(x);
        x = self.conv4(x);
        x = nn.ReLU()(x);
        x = self.conv5(x);
        x = nn.ReLU()(x);

        return x

class MyModel(nn.Module):
    def __init__(self,
                 in_channels,
                 actions_count):
        super(MyModel, self).__init__()

        #64*64*3
        self.block1 = My_NN_Block(in_channels=in_channels, out_channels=16);
        #32*32*16
        self.block2 = My_NN_Block(in_channels=16, out_channels=32);
        #16*16*32
        self.block3 = My_NN_Block(in_channels=32, out_channels=16);
        #8*8*16

        #trying with more linear layers, might do something good
        self.lin1 = nn.Linear(in_features=8*8*16, out_features=256);
        self.lin2 = nn.Linear(in_features=256, out_features=128);
        self.lin3 = nn.Linear(in_features=128, out_features=actions_count);

        self.output_dim = actions_count;
        self.apply(xavier_uniform_init);

    def forward(self, x):

        x = self.block1(x);
        x = self.block2(x);
        x = self.block3(x);
        
        #flatten
        x = x.reshape(x.size(0), -1);
        
        x = self.lin1(x);
        x = nn.ReLU()(x);
        x = self.lin2(x);
        x = nn.ReLU()(x);
        x = self.lin3(x);
        x = nn.ReLU()(x);

        return x