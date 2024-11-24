import torch
from torch import nn
from torch.nn import MultiheadAttention

float_scale = torch.float64


# region ESM3B
class ESM3B_ConvNet_BioInfo_Sub(nn.Module):

    def __init__(self,
                 trial=None):
        super(ESM3B_ConvNet_BioInfo_Sub, self).__init__()
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.7)
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),

            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.LazyLinear(1024)
        )
        self.fc_layer4conv = torch.nn.LazyLinear(1024)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.sub(wild_embedding, mutant_embedding)
        output = self.conv_layer(output)
        output = self.fc_layer4conv(output.reshape(output.shape[0], -1))
        output = torch.concat([output, bioinfo], dim=1)
        output = self.fc_layer(output)
        return output


class ESM3B_ConvNet_BioInfo_Comb(nn.Module):

    def __init__(self,
                 trial=None):
        super(ESM3B_ConvNet_BioInfo_Comb, self).__init__()

        if trial is not None:
            dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        else:
            dropout_rate_linear = 0.5
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )

        self.fc_layer4conv = torch.nn.LazyLinear(1024)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output = self.conv_layer(output)
        output = self.fc_layer4conv(output.reshape(output.shape[0], -1))
        output = torch.concat([output, bioinfo], dim=1)
        output = self.fc_layer(output)
        return output


class ESM3B_AttentionNet_BioInfo_Sub(nn.Module):
    def __init__(self,
                 trial=None):
        super(ESM3B_AttentionNet_BioInfo_Sub, self).__init__()
        if trial is not None:
            dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
            dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
            dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
            num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
            bias = trial.suggest_categorical("bias", [True, False])
        else:
            dropout_rate_attention = 0.5
            dropout_rate_bn = 0.5
            dropout_rate_linear = 0.5
            num_heads = 8
            bias = True
        self.attention_module = MultiheadAttention(embed_dim=2560,
                                                   num_heads=num_heads,
                                                   bias=bias,
                                                   dropout=dropout_rate_attention)
        self.middle_module = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_linear),
            nn.Linear(1024, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, wild_embedding, mutant_embedding, bio):
        output = torch.sub(wild_embedding, mutant_embedding)
        output, attention = self.attention_module(output, output, output)
        output = self.middle_module(output.view(-1, output.shape[-1]))
        output = torch.concat([output.reshape(output.shape[0], -1), bio], dim=1)
        output = self.fc_layer(output.reshape(output.shape[0], -1))
        return output


class ESM3B_AttentionNet_BioInfo_Comb(nn.Module):
    def __init__(self,
                 trial=None):
        super(ESM3B_AttentionNet_BioInfo_Comb, self).__init__()
        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])
        self.attention_layer = MultiheadAttention(embed_dim=5120,
                                                  num_heads=num_heads,
                                                  bias=bias,
                                                  dropout=dropout_rate_attention)
        self.middle_layer = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(5120),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_linear),
            nn.Linear(1024, 128, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Linear(128, 1, dtype=float_scale)
        )

    def forward(self, wild_embedding, mutant_embedding, bio):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output, attention = self.attention_layer(output, output, output)
        output = self.middle_layer(output.view(-1, output.shape[-1]))
        output = torch.concat([output.reshape(output.shape[0], -1), bio], dim=1)
        output = self.fc_layer(output)
        return output


# endregion


# region ESM650M
class ESM650M_ConvNet_BioInfo_Sub(nn.Module):

    def __init__(self,
                 trial=None):
        super(ESM650M_ConvNet_BioInfo_Sub, self).__init__()

        if trial is not None:
            dropout_rate = trial.suggest_float("dropout", 0.2, 0.7)
        else:
            dropout_rate = 0.5

        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),

            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.LazyLinear(1024)
        )
        self.fc_layer4conv = torch.nn.LazyLinear(1024)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.sub(wild_embedding, mutant_embedding)
        output = self.conv_layer(output)
        output = self.fc_layer4conv(output.reshape(output.shape[0], -1))
        output = torch.concat([output, bioinfo], dim=1)
        output = self.fc_layer(output)
        return output


class ESM650M_ConvNet_BioInfo_Comb(nn.Module):

    def __init__(self,
                 trial=None):
        super(ESM650M_ConvNet_BioInfo_Comb, self).__init__()

        if trial is not None:
            dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        else:
            dropout_rate_linear = 0.5
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )

        self.fc_layer4conv = torch.nn.LazyLinear(1024)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output = self.conv_layer(output)
        output = self.fc_layer4conv(output.reshape(output.shape[0], -1))
        output = torch.concat([output, bioinfo], dim=1)
        output = self.fc_layer(output)
        return output


class ESM650M_AttentionNet_BioInfo_Sub(nn.Module):
    def __init__(self,
                 trial=None):
        super(ESM650M_AttentionNet_BioInfo_Sub, self).__init__()
        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])
        self.attention_module = MultiheadAttention(embed_dim=1280,
                                                   num_heads=num_heads,
                                                   bias=bias,
                                                   dropout=dropout_rate_attention)
        self.middle_module = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1280),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_linear),
            nn.Linear(1024, 128, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Linear(128, 1, dtype=float_scale)
        )

    def forward(self, wild_embedding, mutant_embedding, bioInfo):
        output = torch.sub(wild_embedding, mutant_embedding)
        output, attention = self.attention_module(output, output, output)
        output = self.middle_module(output.view(-1, output.shape[-1]))
        output = torch.concat([output.reshape(output.shape[0], -1), bioInfo], dim=1)
        output = self.fc_layer(output.reshape(output.shape[0], -1))
        return output


class ESM650M_AttentionNet_BioInfo_Comb(nn.Module):
    def __init__(self,
                 trial=None):
        super(ESM650M_AttentionNet_BioInfo_Comb, self).__init__()
        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])
        self.attention_layer = MultiheadAttention(embed_dim=2560,
                                                  num_heads=num_heads,
                                                  bias=bias,
                                                  dropout=dropout_rate_attention)
        self.middle_layer = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_linear),
            nn.Linear(1024, 128, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Linear(128, 1, dtype=float_scale)
        )

    def forward(self, wild_embedding, mutant_embedding, bioInfo):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output, attention = self.attention_layer(output, output, output)
        output = self.middle_layer(output.view(-1, output.shape[-1]))
        output = torch.concat([output.reshape(output.shape[0], -1), bioInfo], dim=1)
        output = self.fc_layer(output)
        return output


# endregion


# region ProtBert
class ProtBert_ConvNet_BioInfo_Sub(nn.Module):

    def __init__(self,
                 trial=None):
        super(ProtBert_ConvNet_BioInfo_Sub, self).__init__()

        if trial is not None:
            dropout_rate = trial.suggest_float("dropout", 0.2, 0.7)
        else:
            dropout_rate = 0.5

        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),

            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.LazyLinear(1024)
        )
        self.fc_layer4conv = torch.nn.LazyLinear(1024)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.sub(wild_embedding, mutant_embedding)
        output = self.conv_layer(output)
        output = self.fc_layer4conv(output.reshape(output.shape[0], -1))
        output = torch.concat([output, bioinfo], dim=1)
        output = self.fc_layer(output)
        return output


class ProtBert_ConvNet_BioInfo_Comb(nn.Module):

    def __init__(self,
                 trial=None):
        super(ProtBert_ConvNet_BioInfo_Comb, self).__init__()

        if trial is not None:
            dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        else:
            dropout_rate_linear = 0.5
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )

        self.fc_layer4conv = torch.nn.LazyLinear(1024)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output = self.conv_layer(output)
        output = self.fc_layer4conv(output.reshape(output.shape[0], -1))
        output = torch.concat([output, bioinfo], dim=1)
        output = self.fc_layer(output)
        return output


class ProtBert_AttentionNet_BioInfo_Sub(nn.Module):
    def __init__(self,
                 trial=None):
        super(ProtBert_AttentionNet_BioInfo_Sub, self).__init__()
        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])
        self.attention_module = MultiheadAttention(embed_dim=1024,
                                                   num_heads=num_heads,
                                                   bias=bias,
                                                   dropout=dropout_rate_attention)
        self.middle_module = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_linear),
            nn.Linear(1024, 128, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Linear(128, 1, dtype=float_scale)
        )

    def forward(self, wild_embedding, mutant_embedding, bioInfo):
        output = torch.sub(wild_embedding, mutant_embedding)
        output, attention = self.attention_module(output, output, output)
        output = self.middle_module(output.view(-1, output.shape[-1]))
        output = torch.concat([output.reshape(output.shape[0], -1), bioInfo], dim=1)
        output = self.fc_layer(output.reshape(output.shape[0], -1))
        return output


class ProtBert_AttentionNet_BioInfo_Comb(nn.Module):
    def __init__(self,
                 trial=None):
        super(ProtBert_AttentionNet_BioInfo_Comb, self).__init__()
        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])
        self.attention_layer = MultiheadAttention(embed_dim=2048,
                                                  num_heads=num_heads,
                                                  bias=bias,
                                                  dropout=dropout_rate_attention)
        self.middle_layer = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2048),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_linear),
            nn.Linear(1024, 128, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Linear(128, 1, dtype=float_scale)
        )

    def forward(self, wild_embedding, mutant_embedding, bioInfo):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output, attention = self.attention_layer(output, output, output)
        output = self.middle_layer(output.view(-1, output.shape[-1]))
        output = torch.concat([output.reshape(output.shape[0], -1), bioInfo], dim=1)
        output = self.fc_layer(output)
        return output


# endregion


# region CARP
class CARP_ConvNet_BioInfo_Sub(nn.Module):

    def __init__(self,
                 trial=None):
        super(CARP_ConvNet_BioInfo_Sub, self).__init__()

        if trial is not None:
            dropout_rate = trial.suggest_float("dropout", 0.2, 0.7)
        else:
            dropout_rate = 0.5

        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),

            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.LazyLinear(1024)
        )
        self.fc_layer4conv = torch.nn.LazyLinear(1024)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.sub(wild_embedding, mutant_embedding)
        output = self.conv_layer(output)
        output = self.fc_layer4conv(output.reshape(output.shape[0], -1))
        output = torch.concat([output, bioinfo], dim=1)
        output = self.fc_layer(output)
        return output


class CARP_ConvNet_BioInfo_Comb(nn.Module):

    def __init__(self,
                 trial=None):
        super(CARP_ConvNet_BioInfo_Comb, self).__init__()

        if trial is not None:
            dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        else:
            dropout_rate_linear = 0.5
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )

        self.fc_layer4conv = torch.nn.LazyLinear(1024)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear),
            # torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output = self.conv_layer(output)
        output = self.fc_layer4conv(output.reshape(output.shape[0], -1))
        output = torch.concat([output, bioinfo], dim=1)
        output = self.fc_layer(output)
        return output


class CARP_AttentionNet_BioInfo_Sub(nn.Module):
    def __init__(self,
                 trial=None):
        super(CARP_AttentionNet_BioInfo_Sub, self).__init__()
        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])
        self.attention_module = MultiheadAttention(embed_dim=1280,
                                                   num_heads=num_heads,
                                                   bias=bias,
                                                   dropout=dropout_rate_attention)
        self.middle_module = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1280),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_linear),
            nn.Linear(1024, 128, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Linear(128, 1, dtype=float_scale)
        )

    def forward(self, wild_embedding, mutant_embedding, bioInfo):
        output = torch.sub(wild_embedding, mutant_embedding)
        output, attention = self.attention_module(output, output, output)
        output = self.middle_module(output.view(-1, output.shape[-1]))
        output = torch.concat([output.reshape(output.shape[0], -1), bioInfo], dim=1)
        output = self.fc_layer(output.reshape(output.shape[0], -1))
        return output


class CARP_AttentionNet_BioInfo_Comb(nn.Module):
    def __init__(self,
                 trial=None):
        super(CARP_AttentionNet_BioInfo_Comb, self).__init__()
        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])
        self.attention_layer = MultiheadAttention(embed_dim=2560,
                                                  num_heads=num_heads,
                                                  bias=bias,
                                                  dropout=dropout_rate_attention)
        self.middle_layer = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer = nn.Sequential(
            nn.LazyLinear(1024, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_linear),
            nn.Linear(1024, 128, dtype=float_scale), nn.ReLU(inplace=True),
            nn.Linear(128, 1, dtype=float_scale)
        )

    def forward(self, wild_embedding, mutant_embedding, bioInfo):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output, attention = self.attention_layer(output, output, output)
        output = self.middle_layer(output.view(-1, output.shape[-1]))
        output = torch.concat([output.reshape(output.shape[0], -1), bioInfo], dim=1)
        output = self.fc_layer(output)
        return output


# endregion


if __name__ == "__main__":
    wild, mutant = torch.randn(64, 1, 2560).to(device="cuda", dtype=float_scale), torch.randn(64, 1, 2560).to(
        device="cuda", dtype=float_scale)
    bio = torch.randn(64, 1042).cuda()
    model = ESM3B_AttentionNet_BioInfo_Sub().to(device="cuda", dtype=float_scale)
    result = model(wild, mutant, bio)
