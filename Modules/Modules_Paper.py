from argparse import Namespace
import torch
import math

from .Layer import Conv1d, Linear, Lambda, LayerNorm
from .Gaussian_Upsampler import Gaussian_Upsampler

class XiaoiceSing2(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.encoder = Encoder(self.hp)
        self.variance_block = Variance_Block(self.hp)
        self.decoder = Decoder(self.hp)
        
    def forward(
        self,
        tokens: torch.LongTensor,
        notes: torch.LongTensor,
        durations: torch.LongTensor,
        encoding_lengths: torch.LongTensor,
        genres: torch.LongTensor,
        singers: torch.LongTensor,
        ):
        encodings = self.encoder(
            tokens= tokens,
            notes= notes,
            durations= durations,
            lengths= encoding_lengths,
            genres= genres,
            singers= singers
            )    # [Batch, Enc_d, Feature_t]

        encodings_expand = self.variance_block(
            encodings= encodings,
            encoding_lengths= encoding_lengths,
            durations= durations,  # None when inference
            )
        predictions, log_f0s, voices = self.decoder(
            encodings= encodings_expand,
            lengths= durations.sum(1)
            )

        # Log F0 sum
        # 69.0 -> 81.0 by Midi setting
        max_length = durations.sum(dim= 1).max()
        notes_expand = torch.stack([
            torch.nn.functional.pad(
                note.repeat_interleave(duration, dim= 0),
                pad= [0, max_length - duration.sum()]
                )
            for note, duration in zip(notes, durations)
            ], dim= 0)
        log_f0s_from_note = (440.0 * (2.0 ** ((notes_expand.float() - 81.0) / 12.0))).log() 
        log_f0s_from_note = torch.where(
            log_f0s_from_note > math.log(440.0 * (2.0 ** ((0.0 - 81.0) / 12.0))),
            log_f0s_from_note,
            torch.zeros_like(log_f0s_from_note)
            )
        log_f0s = log_f0s + log_f0s_from_note

        return predictions, log_f0s, voices


class Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size
            )
        self.note_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Notes,
            embedding_dim= self.hp.Encoder.Size
            )
        self.duration_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Durations,
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
        torch.nn.init.xavier_uniform_(self.duration_embedding.weight)
        torch.nn.init.xavier_uniform_(self.genre_embedding.weight)
        torch.nn.init.xavier_uniform_(self.singer_embedding.weight)

        self.positional_encoding = Positional_Encoding(channels= self.hp.Encoder.Size)

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

    def forward(
        self,
        tokens: torch.Tensor,
        notes: torch.Tensor,
        durations: torch.Tensor,
        lengths: torch.Tensor,
        genres: torch.Tensor,
        singers: torch.Tensor
        ):
        x = \
            self.token_embedding(tokens) + \
            self.note_embedding(notes) + \
            self.duration_embedding(durations) + \
            self.genre_embedding(genres).unsqueeze(1) + \
            self.singer_embedding(singers).unsqueeze(1)
        x = x.permute(0, 2, 1)  # [Batch, Enc_d, Enc_t]

        x = self.positional_encoding(x)
        
        for block in self.convfft_blocks:
            x = block(x, lengths)   # [Batch, Enc_d, Enc_t]

        return x

class Decoder(torch.nn.Module): 
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

        self.positional_encoding = Positional_Encoding(channels= self.hp.Encoder.Size)

        self.convfft_blocks = torch.nn.ModuleList([
            ConvFFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Decoder.ConvFFT.Head,
                conv_stack= self.hp.Decoder.ConvFFT.Conv.Stack,
                conv_kernel_size= self.hp.Decoder.ConvFFT.Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Decoder.ConvFFT.FFN.Kernel_Size,
                dropout_rate= self.hp.Decoder.ConvFFT.Dropout_Rate
                )
            for _ in range(self.hp.Decoder.ConvFFT.Stack)    
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
        encodings: torch.Tensor,
        lengths: torch.Tensor
        ):
        x = self.positional_encoding(encodings)
        for block in self.convfft_blocks:
            x = block(x, lengths)   # [Batch, Enc_d, Enc_t]

        predictions, log_f0s, voices = self.linear_projection(x).split_with_sizes([self.feature_size, 1, 1], dim= 1)

        return predictions, log_f0s.squeeze(1), voices.squeeze(1)

class Variance_Block(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.gaussian_upsampler = Gaussian_Upsampler(
            encoding_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Variance_Block.Gaussian_Upsampler.Kernel_Size,
            range_lstm_stack= self.hp.Variance_Block.Gaussian_Upsampler.Range_Predictor.Stack,
            range_dropout_rate= self.hp.Variance_Block.Gaussian_Upsampler.Range_Predictor.Dropout_Rate
            )

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor,
        durations: torch.Tensor= None,  # None when inference
        ):
        alignments = self.gaussian_upsampler(
            encodings= encodings,
            encoding_lengths= encoding_lengths,
            durations= durations
            )   # [Batch, Enc_t, Feature_t]
        
        encodings_expand = encodings @ alignments  # [Batch, Enc_d, Enc_t] @ [Batch, Enc_t, Feature_t] -> [Batch, Enc_d, Feature_t]

        return encodings_expand

class Variance_Predictor(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        lstm_features: int,
        lstm_stack: int,
        dropout_rate: float,
        ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size= in_features,
            hidden_size= lstm_features,
            num_layers= lstm_stack,
            bidirectional= True
            )
        self.lstm_dropout = torch.nn.Dropout(
            p= dropout_rate
            )

        self.projection = torch.nn.Sequential(
            Linear(
                in_features= lstm_features * 2,
                out_features= 1,
                w_init_gain= 'linear'
                ),
            Lambda(lambda x: x.squeeze(2))
            )

    def forward(
        self,
        encodings: torch.Tensor,
        encoding_lengths: torch.Tensor
        ):
        '''
        encodings: [Batch, Enc_d, Enc_t]
        '''
        unpacked_length = encodings.size(2)

        encodings = encodings.permute(2, 0, 1)    # [Enc_t, Batch, Enc_d]        
        if self.training:
            encodings = torch.nn.utils.rnn.pack_padded_sequence(
                encodings,
                encoding_lengths.cpu().numpy(),
                enforce_sorted= False
                )
        
        self.lstm.flatten_parameters()
        encodings = self.lstm(encodings)[0]

        if self.training:
            encodings = torch.nn.utils.rnn.pad_packed_sequence(
                sequence= encodings,
                total_length= unpacked_length
                )[0]
        
        encodings = encodings.permute(1, 0, 2)    # [Batch, Enc_t, Enc_d]
        encodings = self.lstm_dropout(encodings)

        variances = self.projection(encodings)  # [Batch, Enc_t]

        return variances

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

        self.attention = torch.nn.MultiheadAttention(
            embed_dim= channels,
            num_heads= num_head,
            dropout= dropout_rate
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

        attentions = self.attention(
            query= x.permute(2, 0, 1),
            key= x.permute(2, 0, 1),
            value= x.permute(2, 0, 1)
            )[0].permute(1, 2, 0) + x
        attentions = self.attention_norm(attentions * masks)  
        
        convs = x
        for block in self.conv_blocks:
            convs = block(convs, masks)

        x = self.integration_norm(attentions + convs)
        
        # FFN + Dropout + LayerNorm
        x = self.ffn(x, masks)

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
            out_channels= channels * 4,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'relu'
            )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels * 4,
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
            out_channels= channels * 4,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            w_init_gain= 'linear'
            )
        self.norm_0 = LayerNorm(
            num_features= channels * 4,
            )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv1d(
            in_channels= channels * 4,
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


# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/transformer.py
class Positional_Encoding(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.demb = channels
        inv_freq = 1 / (10000 ** (torch.arange(0.0, channels, 2.0) / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        positional_sequence = torch.arange(x.size(2), device= x.device).to(x.dtype)
        sinusoid_inp = self.inv_freq.unsqueeze(1) @ positional_sequence.unsqueeze(0)
        positional_encodings = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim= 0)

        return x + positional_encodings.unsqueeze(0)


def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]