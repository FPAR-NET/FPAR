import torch.nn as nn
import torch.nn.functional as F


# Modulation layer
class SFTLayer(nn.Module):
    def __init__(self, s1, s2, k, p):
        super(SFTLayer, self).__init__()

        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64 * s1, kernel_size=k, stride=s2, padding=p)

        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64 * s1, kernel_size=k, stride=s2, padding=p)

    def forward(self, x):
        # x[0]: fea; x[1]: cond

        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))       # beta
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))       # gamma
        
        #return x[0] * (scale + 1) + shift
        return x[0] * (scale + 1) + shift, scale, shift   # return modulation to visualize it

# Condition layer
def CondNet():
    return nn.Sequential(
        nn.Conv2d(2, 64, 1, 1),  # first should be the number of input channels, in this case two
        nn.LeakyReLU(0.1, True),
        nn.Conv2d(64, 64, 1),
        nn.LeakyReLU(0.1, True),
        nn.Conv2d(64, 64, 1),
        nn.LeakyReLU(0.1, True),
        nn.Conv2d(64, 32, kernel_size=3, padding=1)
    )
