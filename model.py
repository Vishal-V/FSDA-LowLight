import torch
import torch.nn as nn
import torch.nn.functional as F


class lReLU(nn.Module):
    def __init__(self):
        super(lReLU, self).__init__()

    def forward(self, x):
        return torch.max(x * 0.2, x)


class Double_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Double_Conv2d, self).__init__()
        self.double_conv2d = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            lReLU(),
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            lReLU()
        )

    def forward(self, x):
        return self.double_conv2d(x)


class UNetSony(nn.Module):
    def __init__(self):
        super(UNetSony, self).__init__()
        self.conv1 = Double_Conv2d(64, 64)
        self.conv2 = Double_Conv2d(64, 128)
        self.conv3 = Double_Conv2d(128, 256)
        self.conv4 = Double_Conv2d(256, 512)
        self.conv5 = Double_Conv2d(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = Double_Conv2d(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = Double_Conv2d(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = Double_Conv2d(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = Double_Conv2d(128, 64)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, kernel_size=2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, kernel_size=2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, kernel_size=2)

        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4, kernel_size=2)

        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)

        return conv9


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=16, kernel_size=3, padding=1
            ),
            lReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            lReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=out_channel, kernel_size=3, padding=1
            ),
            lReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=32, kernel_size=3, padding=1
            ),
            lReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            lReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_channel, kernel_size=1),
            lReLU(),
        )

    def forward(self, x):

        dc = self.decoder(x)
        out = F.pixel_shuffle(dc, 2)

        return out


class Task_filter(nn.Module):
    def __init__(self):
        super(Task_filter, self).__init__()
        self.en_s = Encoder(4, 64)
        self.en_t = Encoder(4, 64)
        self.unet = UNetSony()
        self.dc = Decoder(64, 12)

    def forward(self, x, source):

        if source:

            for param in self.en_s.parameters():
                param.requires_grad = True
            for param in self.en_t.parameters():
                param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_s(x)
            unet = self.unet(en)
            dc = self.dc(unet)

            return dc

        else:

            for param in self.en_s.parameters():
                param.requires_grad = False
            for param in self.en_t.parameters():
                param.requires_grad = True
            for param in self.unet.parameters():
                param.requires_grad = True
            for param in self.dc.parameters():
                param.requires_grad = True

            en = self.en_t(x)
            unet = self.unet(en)
            dc = self.dc(unet)

            return dc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

class DoubleConv(nn.Module):
    #  Conv--> LReLU **2 (without BN)
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.f(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.f(x)
        return x


class Up(nn.Module):
    # upsample and concat

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up(512, 256)
        self.u2 = Up(256, 128)
        self.u3 = Up(128, 64)
        self.u4 = Up(64, 32)
        self.outc = OutConv(32, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)

        return x