"""
STAFNet -- Spectral-Temporal Attention Fusion Network for Fatigue Detection

Based on: "A Dual-Branch Spectral-Temporal Attention Fusion Network for
EEG-Based Driving Fatigue Detection"
IEEE Transactions on Instrumentation and Measurement, Vol. 75, 2026.
Wu et al. DOI: 10.1109/TIM.2026.3652735

Core idea:
    A dual-branch architecture that jointly models spectral and temporal
    characteristics of time-series signals:
    - Spectral branch: FFT -> frequency band convolution -> SE channel attention
    - Temporal branch: multiscale 1D CNN -> Bi-GRU -> temporal attention
    - Fusion: concatenate both branch outputs -> FC classification

Adaptation notes:
    The original paper uses multi-channel EEG signals (64 channels, 200 Hz,
    5 frequency bands: delta/theta/alpha/beta/gamma). This implementation
    adapts the architecture for ADF (Augmented Deviation Features) input:
    - Input: (B, W, C) where C=3 (drift, diff, local_mean)
    - FFT is applied per channel on the time-domain ADF signal
    - The amplitude spectrum is divided into 5 frequency sub-bands analogous
      to EEG neural rhythms, scaled to the ADF signal length
    - All other modules (band Conv, SE attention, multiscale CNN, Bi-GRU,
      temporal attention, fusion) follow the paper exactly

References:
    Wu, X., Jiang, Z., Fan, C., et al. "A Dual-Branch Spectral-Temporal
    Attention Fusion Network for EEG-Based Driving Fatigue Detection."
    IEEE TIM, vol. 75, 2026.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================== #
#  SE Block (Squeeze-and-Excitation)                                          #
# ========================================================================== #

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (paper Eq. 4-6)

    Applies global average pooling followed by a two-layer bottleneck to
    learn channel-wise importance weights, then reweights the input.

    Args:
        channels: total number of input channels (K * C')
        reduction: channel reduction ratio for the bottleneck
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        """
        Args:
            x: (B, C, 1) where C = K * C'

        Returns:
            (B, C, 1) channel-reweighted features
        """
        B, C, _ = x.shape
        # Squeeze: global average pooling (Eq. 4)
        s = x.mean(dim=-1)                 # (B, C)
        # Excitation: two FC layers with ReLU + Sigmoid (Eq. 5)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))  # (B, C)
        # Scale: channel-wise multiplication (Eq. 6)
        return x * w.unsqueeze(-1)         # (B, C, 1)


# ========================================================================== #
#  Spectral Branch: Frequency-Aware Representation                             #
# ========================================================================== #

class SpectralBranch(nn.Module):
    """Spectral branch: FFT -> frequency band Conv -> SE attention -> FC

    Processes the input signal through:
    1. FFT to obtain amplitude spectrum
    2. Division into K frequency sub-bands
    3. Independent 1D convolution per band (paper Eq. 1)
    4. Adaptive average pooling per band (paper Eq. 2)
    5. Concatenation across bands (paper Eq. 3)
    6. SE channel attention (paper Eq. 4-6)
    7. Flatten + FC to output dimension (paper Eq. 7-8)

    Args:
        in_channels: number of input channels C (e.g., 3 for ADF)
        out_channels_per_band: output channels C' per frequency band
        num_bands: number of frequency sub-bands K (default: 5)
        seq_len: input sequence length W
        output_dim: FC output dimension d (paper: d=2)
        se_reduction: SE bottleneck reduction ratio
    """

    def __init__(self, in_channels=3, out_channels_per_band=8,
                 num_bands=5, seq_len=256, output_dim=2,
                 se_reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_bands = num_bands
        self.out_channels_per_band = out_channels_per_band

        # Number of positive-frequency bins after rfft
        n_fft_bins = seq_len // 2 + 1

        # Divide frequency bins into K approximately equal sub-bands
        band_size = n_fft_bins // num_bands
        self.band_ranges = []
        for i in range(num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < num_bands - 1 else n_fft_bins
            self.band_ranges.append((start, end))

        # Independent 1D convolution per band (paper Eq. 1)
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels_per_band,
                          kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels_per_band),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_bands)
        ])

        # SE attention over concatenated bands (paper Eq. 4-6)
        total_channels = num_bands * out_channels_per_band
        self.se_block = SEBlock(total_channels, reduction=se_reduction)

        # Output FC layer (paper Eq. 8)
        self.fc = nn.Linear(total_channels, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, W, C) input time-domain signal

        Returns:
            (B, output_dim) spectral branch output
        """
        B, W, C = x.shape
        # Transpose to (B, C, W) for FFT along time dimension
        x_t = x.transpose(1, 2)

        # FFT: (B, C, W) -> (B, C, W//2+1) amplitude spectrum
        fft_out = torch.fft.rfft(x_t, dim=-1)
        amplitude = fft_out.abs()            # (B, C, W//2+1)

        band_features = []
        for i, (start, end) in enumerate(self.band_ranges):
            # Extract frequency band: (B, C, L_band)
            band_amp = amplitude[:, :, start:end]
            # Frequency band convolution + BN + ReLU (Eq. 1)
            h = self.band_convs[i](band_amp)  # (B, C', L_band)
            # Adaptive average pooling (Eq. 2)
            z = F.adaptive_avg_pool1d(h, 1)   # (B, C', 1)
            band_features.append(z)

        # Concatenate all bands (Eq. 3)
        z_freq = torch.cat(band_features, dim=1)  # (B, K*C', 1)

        # SE channel attention (Eq. 4-6)
        z_se = self.se_block(z_freq)               # (B, K*C', 1)

        # Flatten + FC (Eq. 7-8)
        z_flat = z_se.squeeze(-1)                   # (B, K*C')
        o_f = self.fc(z_flat)                       # (B, output_dim)

        return o_f


# ========================================================================== #
#  Temporal Branch: Multiscale Sequential Modeling                             #
# ========================================================================== #

class TemporalBranch(nn.Module):
    """Temporal branch: multiscale 1D CNN -> Bi-GRU -> temporal attention -> FC

    Processes the input signal through:
    1. Parallel 1D convolutions with kernel sizes {3, 5, 7} (paper Eq. 9)
    2. Concatenation + 1x1 fusion convolution (paper Eq. 10)
    3. Bidirectional GRU for long-range dependencies (paper Eq. 11)
    4. Temporal attention for weighted feature fusion (paper Eq. 12)
    5. Two FC layers for classification (paper Eq. 13-14)

    Args:
        in_channels: number of input channels C (e.g., 3 for ADF)
        conv_channels: output channels per convolution scale C'
        gru_hidden: GRU hidden size H (unidirectional, Bi-GRU outputs 2H)
        gru_layers: number of GRU layers
        output_dim: FC output dimension (paper: 2)
        dropout: dropout rate for FC layers
    """

    def __init__(self, in_channels=3, conv_channels=16,
                 gru_hidden=64, gru_layers=1,
                 output_dim=2, dropout=0.1):
        super().__init__()
        self.gru_hidden = gru_hidden

        # Multiscale 1D convolutions (paper Eq. 9, kernel sizes 3, 5, 7)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )

        # 1x1 fusion convolution (paper Eq. 10)
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(conv_channels * 3, conv_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Bidirectional GRU (paper Eq. 11)
        self.bi_gru = nn.GRU(
            conv_channels, gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Temporal attention (paper Eq. 12)
        self.attn_weight = nn.Linear(2 * gru_hidden, 1)

        # FC classification layers (paper Eq. 13-14)
        self.fc1 = nn.Linear(2 * gru_hidden, 2 * gru_hidden)
        self.fc2 = nn.Linear(2 * gru_hidden, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, W, C) input time-domain signal

        Returns:
            (B, output_dim) temporal branch output
        """
        # Transpose to (B, C, W) for Conv1d
        x_t = x.transpose(1, 2)

        # Multiscale convolutions (Eq. 9)
        f3 = self.conv3(x_t)   # (B, conv_channels, W)
        f5 = self.conv5(x_t)   # (B, conv_channels, W)
        f7 = self.conv7(x_t)   # (B, conv_channels, W)

        # Concatenate + 1x1 fusion (Eq. 10)
        f_cat = torch.cat([f3, f5, f7], dim=1)  # (B, 3*conv_channels, W)
        f_fused = self.fusion_conv(f_cat)        # (B, conv_channels, W)

        # Transpose for GRU: (B, conv_channels, W) -> (B, W, conv_channels)
        f_seq = f_fused.transpose(1, 2)

        # Bi-GRU (Eq. 11)
        H, _ = self.bi_gru(f_seq)  # (B, W, 2H)

        # Temporal attention (Eq. 12)
        # alpha_t = softmax(w^T * H_t) over time steps
        scores = self.attn_weight(H).squeeze(-1)  # (B, W)
        alpha = F.softmax(scores, dim=1)           # (B, W)
        z_att = (H * alpha.unsqueeze(-1)).sum(dim=1)  # (B, 2H)

        # FC layers (Eq. 13-14)
        z1 = F.relu(self.fc1(z_att))   # (B, 2H)
        z1 = self.dropout(z1)
        o_t = self.fc2(z1)             # (B, output_dim)

        return o_t


# ========================================================================== #
#  STAFNet Classifier                                                         #
# ========================================================================== #

class STAFNetClassifier(nn.Module):
    """STAFNet: Spectral-Temporal Attention Fusion Network

    Dual-branch architecture for fatigue detection combining frequency-aware
    spectral representation with multiscale temporal modeling.

    Architecture (following paper Fig. 1):
        Input (B, W, C)
            |
            +---> Spectral Branch -> o_f (B, d)
            |         FFT -> Band Conv x K -> SE Attention -> FC
            |
            +---> Temporal Branch -> o_t (B, d)
            |         Multiscale CNN -> Bi-GRU -> Temporal Attention -> FC
            |
            +---> Fusion: Concat(o_f, o_t) -> FC -> softmax -> predictions
                      (B, 2d)               (B, num_classes)

    Args:
        input_size: number of input channels C (ADF: 3, single: 1)
        seq_len: input sequence length W (default: 256)
        num_classes: number of output classes (default: 2)
        spectral_channels: output channels C' per frequency band (paper: 8)
        num_bands: number of frequency sub-bands K (paper: 5)
        se_reduction: SE block reduction ratio (paper: 4)
        temporal_channels: multiscale conv output channels C' (paper: 16)
        gru_hidden: Bi-GRU unidirectional hidden size H (paper: 64)
        gru_layers: number of GRU layers (paper: 1)
        branch_output_dim: per-branch FC output dim d (paper: 2)
        dropout: dropout rate (paper: uses weight_decay instead)
    """

    def __init__(self, input_size=3, seq_len=256, num_classes=2,
                 spectral_channels=8, num_bands=5, se_reduction=4,
                 temporal_channels=16, gru_hidden=64, gru_layers=1,
                 branch_output_dim=2, dropout=0.1):
        super().__init__()

        # Spectral branch (paper Section III-B)
        self.spectral_branch = SpectralBranch(
            in_channels=input_size,
            out_channels_per_band=spectral_channels,
            num_bands=num_bands,
            seq_len=seq_len,
            output_dim=branch_output_dim,
            se_reduction=se_reduction,
        )

        # Temporal branch (paper Section III-C)
        self.temporal_branch = TemporalBranch(
            in_channels=input_size,
            conv_channels=temporal_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            output_dim=branch_output_dim,
            dropout=dropout,
        )

        # Fusion layer (paper Section III-D, Eq. 15-16)
        # o_fusion = Concat(o_f, o_t) in R^{2d}
        # O_final = W_fuse * o_fusion + b_fuse in R^{num_classes}
        self.fusion_fc = nn.Linear(branch_output_dim * 2, num_classes)

    def forward(self, x):
        """Forward pass through dual-branch architecture

        Args:
            x: (B, W, C) input time-domain signal (ADF features)

        Returns:
            (B, num_classes) logits
        """
        # Spectral branch output (Eq. 8)
        o_f = self.spectral_branch(x)     # (B, d)

        # Temporal branch output (Eq. 14)
        o_t = self.temporal_branch(x)     # (B, d)

        # Fusion (Eq. 15)
        o_fusion = torch.cat([o_f, o_t], dim=-1)  # (B, 2d)

        # Classification (Eq. 16)
        logits = self.fusion_fc(o_fusion)           # (B, num_classes)

        return logits
