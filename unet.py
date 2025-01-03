import torch
import torch.nn as nn
import numpy as np

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs

class zs_model(nn.Module):
    """
    may need to add something like dimensions in here? so we can build the unet at the appropriate size.
    """
    def __init__(self):
        super().__init__()

        self.unet = build_unet()

    def forward(self, kspace):

        # take IFT to make it image space
        im_space = torch.fft.ifftshift( torch.fft.ifftn( torch.fft.fftshift( kspace ) ) )

        im_space_r = torch.view_as_real(im_space)
        n = im_space_r.shape[-3]
        im_space_stack = torch.cat((im_space_r[..., 0], im_space_r[..., 1]), dim=2)

        # run unet
        post_unet = self.unet(im_space_stack)

        # split back into real/imaginary

        post_unet_r = post_unet[..., 1:n+1, :]
        post_unet_im = post_unet[..., n:, :]

        post_unet = torch.stack((post_unet_r, post_unet_im), dim=-1)
        post_unet = torch.view_as_complex(post_unet)

        # FT back to k-space
        kspace_out = torch.fft.fftshift( torch.fft.fftn( torch.fft.ifftshift( post_unet ) ) )

        return kspace_out
    


if __name__ == "__main__":
    # x = torch.randn((2, 3, 512, 512))
    x = torch.randn((1, 1, 640, 320)) + 1j*torch.randn((1,1,640,320))
    print(x.dtype)
    # f = build_unet()
    # y = f(x)
    # print(y.shape)

    f2 = zs_model()
    y2 = f2(x)
    print(y2.shape)
    print(y2.dtype)
