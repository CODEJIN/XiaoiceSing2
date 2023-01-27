# From Exp54
import torch
from typing import Tuple

from .Layer import Conv1d, Conv2d


class Discriminator(torch.nn.Module):
    def __init__(self, hyper_parameters):
        super().__init__()
        self.hp = hyper_parameters

        if self.hp.Feature_Type == 'Spectrogram':
            feature_size = self.hp.Sound.N_FFT // 2 + 1
        elif self.hp.Feature_Type == 'Mel':
            feature_size = self.hp.Sound.Mel_Dim
        else:
            raise ValueError('Unknown feature type: {}'.format(self.hp.Feature_Type))

        self.segment_disrcriminators = torch.nn.ModuleList([
            Segment_Discriminator(
                in_channels= feature_size,
                calc_channels= self.hp.Discriminator.Segment.Channels,
                kernel_size= self.hp.Discriminator.Segment.Kernel_Size,
                conv_stack= self.hp.Discriminator.Segment.Stack,
                segment_size= segment_size
                )
            for segment_size in self.hp.Discriminator.Segment.Segment_Size
            ])
            
        self.detail_disrcriminators = torch.nn.ModuleList([
            Detail_Discriminator(
                calc_channels= self.hp.Discriminator.Detail.Channels,
                kernel_size= self.hp.Discriminator.Detail.Kernel_Size,
                downsample_stack= self.hp.Discriminator.Detail.Downsample_Stack,
                conv_stack= self.hp.Discriminator.Detail.Conv_Stack,
                frequency_index= (frequency_index, frequency_index + feature_size // 2)
                )
            for frequency_index in range(0, feature_size // 2 + 1, feature_size // 4)
            ])

    def forward(self, features: torch.Tensor):
        segment_discriminations_list = []
        segment_feature_maps_list = []
        for discriminator in self.segment_disrcriminators:
            discriminations, feature_maps_list = discriminator(features)
            segment_discriminations_list.append(discriminations)
            segment_feature_maps_list.extend(feature_maps_list)

        detail_discriminations_list = []
        detail_feature_maps_list = []
        for discriminator in self.detail_disrcriminators:
            discriminations, feature_maps_list = discriminator(features)
            detail_discriminations_list.append(discriminations)
            detail_feature_maps_list.extend(feature_maps_list)

        return \
            segment_discriminations_list, segment_feature_maps_list, \
            detail_discriminations_list, detail_feature_maps_list

class Segment_Discriminator(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        calc_channels: int,
        kernel_size: int,
        conv_stack: int,
        segment_size: int,
        leaky_relu_slope: float= 0.2
        ):
        super().__init__()
        self.segment_size = segment_size

        self.conv_blocks = torch.nn.ModuleList()

        previous_channels = in_channels
        for _ in range(conv_stack):
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv1d(
                    in_channels= previous_channels,
                    out_channels= calc_channels,
                    kernel_size= kernel_size,
                    padding= (kernel_size - 1) // 2,
                    w_init_gain= 'leaky_relu'
                    )),
                torch.nn.LeakyReLU(
                    negative_slope= leaky_relu_slope,
                    ),
                )
            self.conv_blocks.append(block)
            previous_channels = calc_channels

        self.conv_blocks.append(torch.nn.utils.weight_norm(Conv1d(
            in_channels= calc_channels,
            out_channels= 1,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2
            )))

    def forward(self, features: torch.Tensor):
        '''
        features: [Batch, Feature_d, Feature_t]
        '''
        offsets = (torch.rand_like(features[:, 0, 0]) * (features.size(2) - self.segment_size)).clamp(0).long()
        x = torch.stack([
            feature[:, offset:offset + self.segment_size]
            for feature, offset in zip(features, offsets)
            ], dim= 0)

        feature_maps_list = []
        for block in self.conv_blocks:
            x = block(x)
            feature_maps_list.append(x)

        return x.squeeze(1), feature_maps_list

class Detail_Discriminator(torch.nn.Module):
    def __init__(
        self,
        calc_channels: int,
        kernel_size: int,
        downsample_stack: int,
        conv_stack: int,
        frequency_index: Tuple[int, int],
        leaky_relu_slope: float= 0.2
        ):
        super().__init__()
        self.frequency_index = frequency_index

        self.conv_blocks = torch.nn.ModuleList()
        self.conv_blocks.append(torch.nn.Sequential(
            torch.nn.utils.weight_norm(Conv2d(
                in_channels= 1,
                out_channels= calc_channels,
                kernel_size= kernel_size,
                padding= (kernel_size - 1) // 2,
                w_init_gain= 'leaky_relu'
                )),
            torch.nn.LeakyReLU(
                negative_slope= leaky_relu_slope,
                ),
            ))
        
        for _ in range(downsample_stack):
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv2d(
                    in_channels= calc_channels,
                    out_channels= calc_channels,
                    kernel_size= kernel_size,
                    padding= (kernel_size - 1) // 2,
                    stride= 2,
                    w_init_gain= 'leaky_relu'
                    )),
                torch.nn.LeakyReLU(
                    negative_slope= leaky_relu_slope,
                    ),
                )
            self.conv_blocks.append(block)

        for _ in range(conv_stack):
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv2d(
                    in_channels= calc_channels,
                    out_channels= calc_channels,
                    kernel_size= (1, kernel_size),
                    padding= (0, (kernel_size - 1) // 2),
                    w_init_gain= 'leaky_relu'
                    )),
                torch.nn.LeakyReLU(
                    negative_slope= leaky_relu_slope,
                    ),
                )
            self.conv_blocks.append(block)

        self.conv_blocks.append(torch.nn.utils.weight_norm(Conv2d(
            in_channels= calc_channels,
            out_channels= 1,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2
            )))

    def forward(self, features: torch.Tensor):
        '''
        features: [Batch, Feature_d, Feature_t]
        '''
        x = features[:, self.frequency_index[0]:self.frequency_index[1]].unsqueeze(1)   # [Batch, 1, Sub_Feature_d, Feature_t]

        feature_maps_list = []
        for block in self.conv_blocks:
            x = block(x)
            feature_maps_list.append(x)

        return x.squeeze(1), feature_maps_list

class R1_Regulator(torch.nn.Module):
    def forward(
        self,
        segment_discriminations_list: torch.Tensor,
        detail_discriminations_list: torch.Tensor,
        features: torch.Tensor
        ):
        x = torch.autograd.grad(
            outputs= [
                discriminations.sum()
                for discriminations in segment_discriminations_list + detail_discriminations_list
                ],
            inputs= features,
            create_graph= True,
            retain_graph= True,
            only_inputs= True
            )[0].pow(2)
        x = (x.view(segment_discriminations_list[0].size(0), -1).norm(2, dim=1) ** 2).mean()

        return x


