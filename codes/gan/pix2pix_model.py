import torch
import torch.nn as nn


def _block(in_channels, out_channels, down_or_up, relu='relu'):
    assert down_or_up in ('down', 'up')
    assert relu == 'relu' or (isinstance(relu, float) and 0. <= relu < 1.)
    ConvModule = nn.Conv2d if down_or_up == 'down' else nn.ConvTranspose2d
    ReluModule, relu_args = (nn.ReLU, ()) if relu == 'relu' else (nn.LeakyReLU, (0.2,))
    return nn.Sequential(
        ConvModule(in_channels, out_channels, 4, 2, True),
        nn.BatchNorm2d(out_channels),
        ReluModule(*relu_args),
    )


# generator
class UNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.down1 = _block(3, 64, 'down')
        self.down2 = _block(64, 128, 'down')
        self.down3 = _block(128, 256, 'down')
        self.down4 = _block(256, 256, 'down')
        self.down5 = _block(256, 256, 'down')
        self.bottleneck_down = _block(256, 256, 'down')
        self.bottleneck_up = _block(256, 256, 'up')
        self.up5 = _block(2 * 256, 256, 'up')
        self.up4 = _block(2 * 256, 256, 'up')
        self.up3 = _block(2 * 256, 128, 'up')
        self.up2 = _block(2 * 128, 64, 'up')
        self.up1 = _block(2 * 64, 3, 'up')
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)  # 256, 8 * 8
        out = self.bottleneck_up(self.bottleneck_down(x5))  # 256, 8*8 -> 4*4 -> 8*8
        out = self.up5(torch.cat([out, x5], dim=1))  # 256, 8 * 8
        out = self.up4(torch.cat([out, x4], dim=1))  # 256, 16 * 16
        out = self.up3(torch.cat([out, x3], dim=1))  # 128, 32 * 32
        out = self.up2(torch.cat([out, x2], dim=1))
        out = self.up1(torch.cat([out, x1], dim=1))
        out = self.tanh(out)
        return out


# discriminator
class Images2Scaler(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.down1 = _block(2 * 3, 64, 'down', 0.2)
        self.down2 = _block(64, 128, 'down', 0.2)
        self.down3 = _block(128, 256, 'down', 0.2)
        self.down4 = _block(256, 256, 'down', 0.2)
        self.down5 = _block(256, 256, 'down', 0.2)
        self.down6 = _block(256, 256, 'down')  # (32, 256, 4, 4)
        self.final_conv = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        a = torch.cat([x, y], dim=1)
        a = self.down1(a)
        a = self.down2(a)
        a = self.down3(a)
        a = self.down4(a)
        a = self.down5(a)
        a = self.down6(a)
        a = self.final_conv(a)
        a = self.sigmoid(a)  # bz, 1, 1, 1
        return a.view(-1)


if __name__ == '__main__':
    IMG_SIZE = 256
    x = torch.randn((32, 3, IMG_SIZE, IMG_SIZE))
    print(x.shape)

    unet = UNet(None)
    unet(x)
