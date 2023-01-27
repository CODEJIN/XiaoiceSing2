from argparse import Namespace
import torch
import math

from .Layer import Conv1d, LayerNorm

class XiaoiceSing2(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.encoder = Encoder(self.hp)
        
    def forward(
        self,
        tokens: torch.LongTensor,
        notes: torch.LongTensor,
        lengths: torch.LongTensor,
        genres: torch.LongTensor,
        singers: torch.LongTensor,
        ):
        predictions, log_f0s, voices = self.encoder(
            tokens= tokens,
            notes= notes,
            lengths= lengths,
            genres= genres,
            singers= singers
            )    # [Batch, Enc_d, Feature_t]

        return predictions, log_f0s, voices


class Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        if self.hp.Feature_Type == 'Mel':
            self.feature_size = self.hp.Sound.Mel_Dim
        elif self.hp.Feature_Type == 'Spectrogram':
            self.feature_size = self.hp.Sound.N_FFT // 2 + 1

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size
            )
        self.note_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Notes,
            embedding_dim= self.hp.Encoder.Size
            )
        self.genre_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Genres,
            embedding_dim= self.hp.Encoder.Size,
            )
        self.singer_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Singers,
            embedding_dim= self.hp.Encoder.Size,
            )
        torch.nn.init.xavier_uniform_(self.token_embedding.weight)
        torch.nn.init.xavier_uniform_(self.note_embedding.weight)
        torch.nn.init.xavier_uniform_(self.genre_embedding.weight)
        torch.nn.init.xavier_uniform_(self.singer_embedding.weight)

        self.convfft_blocks = torch.nn.ModuleList([
            ConvFFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.ConvFFT.Head,
                conv_stack= self.hp.Encoder.ConvFFT.Conv.Stack,
                conv_kernel_size= self.hp.Encoder.ConvFFT.Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Encoder.ConvFFT.FFN.Kernel_Size,
                dropout_rate= self.hp.Encoder.ConvFFT.Dropout_Rate
                )
            for _ in range(self.hp.Encoder.ConvFFT.Stack)    
            ])

        self.linear_projection = Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.feature_size + 2, # Feature + V/UV + Log F0
            kernel_size= 1,
            bias= True,
            w_init_gain= 'linear'     
            )

    def forward(
        self,
        tokens: torch.Tensor,
        notes: torch.Tensor,
        lengths: torch.Tensor,
        genres: torch.Tensor,
        singers: torch.Tensor
        ):
        x = \
            self.token_embedding(tokens) + \
            self.note_embedding(notes) + \
            self.genre_embedding(genres).unsqueeze(1) + \
            self.singer_embedding(singers).unsqueeze(1)
        x = x.permute(0, 2, 1)  # [Batch, Enc_d, Enc_t]

        for block in self.convfft_blocks:
            x = block(x, lengths)   # [Batch, Enc_d, Enc_t]

        predictions, log_f0s, voices = self.linear_projection(x).split_with_sizes([self.feature_size, 1, 1], dim= 1)
        log_f0s, voices = log_f0s.squeeze(1), voices.squeeze(1)
        
        # Log F0 sum
        # 69.0 -> 81.0 by Midi setting        
        log_f0s_from_note = (440.0 * (2.0 ** ((notes.float() - 81.0) / 12.0))).log() 
        log_f0s_from_note = torch.where(
            log_f0s_from_note > math.log(440.0 * (2.0 ** ((0.0 - 81.0) / 12.0))),
            log_f0s_from_note,
            torch.zeros_like(log_f0s_from_note)
            )
        log_f0s = log_f0s + log_f0s_from_note

        return predictions, log_f0s, voices

class ConvFFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,
        conv_stack: int,
        conv_kernel_size: int,
        ffn_kernel_size: int,
        dropout_rate: float= 0.1
        ) -> None:
        super().__init__()

        self.attention = LinearAttention(
            channels= channels,
            calc_channels= channels,
            num_heads= num_head,
            dropout_rate= dropout_rate
            )
        self.attention_norm = LayerNorm(
            num_features= channels,
            )

        self.conv_blocks = torch.nn.ModuleList([
            Conv_Block(
                channels= channels,
                kernel_size= conv_kernel_size,
                dropout_rate= dropout_rate
                )
            for _ in range(conv_stack)
            ])
        self.integration_norm = LayerNorm(
            num_features= channels,
            )
        
        self.ffn = FFN(
            channels= channels,
            kernel_size= ffn_kernel_size,
            dropout_rate= dropout_rate
            )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        masks = (~Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())).unsqueeze(1).float()   # float mask

        # Attention + Dropout + LayerNorm
        attentions = self.attention(x)
        
        convs = x
        for block in self.conv_blocks:
            convs = block(convs, masks)

        x = self.integration_norm(attentions + convs)
        
        # FFN + Dropout + LayerNorm
        x = self.ffn(x, masks)

        return x * masks

class LinearAttention(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        num_heads: int,
        dropout_rate: float= 0.1,
        use_scale: bool= True,
        use_residual: bool= True,
        use_norm: bool= True
        ):
        super().__init__()
        assert calc_channels % num_heads == 0
        self.calc_channels = calc_channels
        self.num_heads = num_heads
        self.use_scale = use_scale
        self.use_residual = use_residual
        self.use_norm = use_norm

        self.prenet = Conv1d(
            in_channels= channels,
            out_channels= calc_channels * 3,
            kernel_size= 1,
            bias=False,
            w_init_gain= 'linear'
            )
        self.projection = Conv1d(
            in_channels= calc_channels,
            out_channels= channels,
            kernel_size= 1,
            w_init_gain= 'linear'
            )
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        
        if use_scale:
            self.scale = torch.nn.Parameter(torch.zeros(1))

        if use_norm:
            self.norm = LayerNorm(num_features= channels)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        '''
        x: [Batch, Enc_d, Enc_t]
        '''
        residuals = x

        x = self.prenet(x)  # [Batch, Calc_d * 3, Enc_t]        
        x = x.view(x.size(0), self.num_heads, x.size(1) // self.num_heads, x.size(2))    # [Batch, Head, Calc_d // Head * 3, Enc_t]
        queries, keys, values = x.chunk(chunks= 3, dim= 2)  # [Batch, Head, Calc_d // Head, Enc_t] * 3
        keys = (keys + 1e-5).softmax(dim= 3)

        contexts = keys @ values.permute(0, 1, 3, 2)   # [Batch, Head, Calc_d // Head, Calc_d // Head]
        contexts = contexts.permute(0, 1, 3, 2) @ queries   # [Batch, Head, Calc_d // Head, Enc_t]
        contexts = contexts.view(contexts.size(0), contexts.size(1) * contexts.size(2), contexts.size(3))   # [Batch, Calc_d, Enc_t]
        contexts = self.projection(contexts)    # [Batch, Enc_d, Enc_t]

        if self.use_scale:
            contexts = self.scale * contexts

        contexts = self.dropout(contexts)

        if self.use_residual:
            contexts = contexts + residuals

        if self.use_norm:
            contexts = self.norm(contexts)

        return contexts

class Conv_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout_rate: float= 0.1
        ) -> None:
        super().__init__()

        self.conv_0 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        self.norm_0 = LayerNorm(
            num_features= channels,
            )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        self.norm_1 = LayerNorm(
            num_features= channels,
            )

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        residuals = x

        x = self.conv_0(x * masks)
        x = self.norm_0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_1(x * masks)
        x = self.norm_1(x + residuals)
        
        return x * masks

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout_rate: float= 0.1,
        ) -> None:
        super().__init__()
        self.conv_0 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'relu'
            )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        self.norm = LayerNorm(
            num_features= channels,
            )
        
    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        residuals = x

        x = self.conv_0(x * masks)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_1(x * masks)
        x = self.dropout(x)
        x = self.norm(x + residuals)

        return x * masks


def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]