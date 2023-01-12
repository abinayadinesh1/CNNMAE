import torch
from torch import nn

import numpy as np

import math


# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# does not contain trainable params so no need for Module declaration
def drop_path(x, keep_prob=1.0):
    # dim 0 is batch size, 1, 1, 1
    mask = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep_prob)
    x = x / keep_prob

    return x*mask


# creates a mask which is applied post-convolution to prevent info leakage
def mask_creator(input_batch, kernel_size, out_channels, padding="not-same", stride=1):
    kernel_y_span = math.ceil((input_batch.size()[2]-kernel_size+1)/stride)
    kernel_x_span = math.ceil((input_batch.size()[3]-kernel_size+1)/stride)

    if padding == "same":
        output = torch.zeros((input_batch.size()[0], out_channels, kernel_y_span+kernel_size-1, 
            kernel_x_span+kernel_size-1))
    else:
        output = output = torch.ones((input_batch.size()[0], out_channels, kernel_y_span, kernel_x_span))

    if padding == "same":
        for kernel_y in range(0, kernel_y_span):
            for kernel_x in range(0, kernel_x_span):
                if input_batch[:,:,kernel_y:kernel_y+kernel_size,kernel_x:kernel_x+kernel_size].all():
                    output[:,:,int((kernel_size-1)/2)+kernel_y,int((kernel_size-1)/2)+kernel_x] = 1
    else:
        for kernel_y in range(kernel_y_span):
            for kernel_x in range(kernel_x_span):
                if not input_batch[:,:,kernel_y*stride:(kernel_y*stride)+kernel_size,kernel_x*stride:(kernel_x*stride)+kernel_size].all():
                    output[:,:,kernel_y,kernel_x] = 0
    
    return output


# cuz pytorch is bad
class DepthwiseConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, padding="same")
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, padding="same")

    def forward(self, x):
        x = self.depthwise_conv(x)

        return self.pointwise_conv(x)


# global response normalization to reduce feature collapse
class GRN(nn.Module):
    def __init__(self, batch_size, in_channels):
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros((batch_size, in_channels*4, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((batch_size, in_channels*4, 1, 1)))

        
    def forward(self, x):
        original = x

        # global feature aggregation with l2 matrix norm
        x = torch.linalg.matrix_norm(x, ord=2, dim=(2, 3))

        # feature normalization (standard divisive normalization)
        x = x/torch.sum(x)

        # feature calibration
        x = original*x[:, :, None, None]

        #adding trainable paramaters for learned GRN
        x = self.gamma*x + self.beta + original

        return x


# to downsample + make image dimensions smaller
class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims):
        super().__init__()

        self.norm = nn.LayerNorm((in_channels, input_dims[0], input_dims[1]), eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, 2, stride=2)

        
    def forward(self, x):
        x = self.norm(x)
        return self.conv(x)


# block used for masked autoencoder
class ConvNeXtEncoderBlock(nn.Module):
    def __init__(self, channels, kernel_size, batch_size, input_dims, keep_prob=1.0):
        super().__init__()

        self.keep_prob = keep_prob
        
        # depthwise convolution layer consists of depthwise convolution +
        #   pointwise convolution operations on the outputs of depthwise 
        #   convolution
        self.conv = DepthwiseConvolution(channels, channels, kernel_size)
        self.norm = nn.LayerNorm((channels, input_dims[0], input_dims[1]), eps=1e-6)
        self.linear1 = nn.Conv2d(channels, channels*4, 1)
        self.gelu = GELU()
        self.grn = GRN(batch_size, channels)
        self.linear2 = nn.Conv2d(channels*4, channels, 1)


    def forward(self, x):
        # order of computations: depthwise conv, layer norm, 
        #   lin1, gelu, grn, lin2, drop path, skip
        original = x

        x = self.conv(x)
        x = self.norm(x)

        x = self.linear1(x)
        x = self.gelu(x)

        x = self.grn(x)

        x = self.linear2(x)

        x = original+drop_path(x, self.keep_prob)

        return x


# block used for masked autoencoder
class ConvNeXtDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_size, input_dims, stride, keep_prob=1.0):
        super().__init__()

        self.keep_prob = keep_prob
        
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.norm = nn.LayerNorm((out_channels, input_dims[0], input_dims[1]), eps=1e-6)
        self.linear1 = nn.Conv2d(out_channels, out_channels*4, 1)
        self.gelu = GELU()
        self.grn = GRN(batch_size, out_channels)
        self.linear2 = nn.Conv2d(out_channels*4, out_channels, 1)


    def forward(self, x):
        # order of computations: depthwise conv, layer norm, 
        #   lin1, gelu, grn, lin2, drop path

        x = self.conv(x)
        x = self.norm(x)

        x = self.linear1(x)
        x = self.gelu(x)

        x = self.grn(x)

        x = self.linear2(x)

        x = drop_path(x, self.keep_prob)

        return x


# creating fully convolutional masked autoencoder
class FCMAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.channel_list = config.channel_list


        # ENCODER

        curr_input_dims = (math.ceil((config.input_dims[1]-3)/2), math.ceil((config.input_dims[1]-3)/2))

        self.stem_conv = nn.Conv2d(config.channel_list[0], config.channel_list[1], kernel_size=4, stride=2)
        self.stem_norm = nn.LayerNorm((config.channel_list[1], curr_input_dims[0], curr_input_dims[1]), eps=1e-6)

        self.downsample_layers = []
        self.convnext_blocks = []
        for i in range(1, config.num_blocks):
            self.downsample_layers += [DownsampleLayer(config.channel_list[i], config.channel_list[i+1], curr_input_dims)]
            curr_input_dims = (math.ceil((curr_input_dims[0]-2)/2), math.ceil((curr_input_dims[1]-2)/2))
            self.convnext_blocks += [ConvNeXtEncoderBlock(config.channel_list[i+1], (7, 7), config.batch_size, curr_input_dims)]

        self.downsample_layers = nn.ModuleList(self.downsample_layers)
        self.convnext_blocks = nn.ModuleList(self.convnext_blocks)


        # DECODER

        self.decoder_block = ConvNeXtDecoderBlock(config.channel_list[-1], 3, (16, 16), config.batch_size, (256, 256), 8)
        self.final = nn.Tanh()

        
    def forward(self, x):
        # encoder pass
        x = self.stem_conv(x)
        x = self.stem_norm(x)

        for i in range(len(self.downsample_layers)):
            downsample_mask = mask_creator(x, 2, self.channel_list[i+2], stride=2)
            x = self.downsample_layers[i](x)
            x = x*downsample_mask

            convnext_block_mask = mask_creator(x, 7, self.channel_list[i+2], padding="same", stride=1)
            x = self.convnext_blocks[i](x)
            x = x*convnext_block_mask
        
        # decoder pass
        x = self.decoder_block(x)

        return self.final(x)