import torch
from torch import nn
from torch.nn import MultiheadAttention

float_scale = torch.float64


# region ESM3B
class ESM3B_ConvNet_BioInfo_Sub(nn.Module):

    def __init__(self,
                 trial=None):
        super(ESM3B_ConvNet_BioInfo_Sub, self).__init__()
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.LazyLinear(1024)
        )
        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        # region concating
        conc = torch.sub(wild_embedding, mutant_embedding)
        conc_out = self.conv_layer(conc)
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), bioinfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out = self.conv_layer(wild_embedding)
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out = self.conv_layer(mutant_embedding)
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


class ESM3B_ConvNet_BioInfo_Comb(nn.Module):

    def __init__(self,
                 trial=None):
        super(ESM3B_ConvNet_BioInfo_Comb, self).__init__()

        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        # region concating
        conc = torch.concat([wild_embedding, mutant_embedding], dim=2)
        conc_out = self.conv_layer(conc)
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), bioinfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out = self.conv_layer(wild_embedding)
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out = self.conv_layer(mutant_embedding)
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


class ESM3B_AttentionNet_BioInfo_Sub(nn.Module):
    def __init__(self,
                 trial=None):
        super(ESM3B_AttentionNet_BioInfo_Sub, self).__init__()

        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])

        self.attention_layer01 = MultiheadAttention(embed_dim=2560,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer01 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer02 = MultiheadAttention(embed_dim=2560,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer02 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer03 = MultiheadAttention(embed_dim=2560,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer03 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, BioInfo):
        # region concating
        subt = torch.sub(wild_embedding, mutant_embedding)
        subt_out, _ = self.attention_layer01(subt, subt, subt)
        subt_out = self.middle_layer01(subt_out.view(-1, subt_out.shape[-1]))
        subt_out = torch.concat([subt_out.reshape(subt_out.shape[0], -1), BioInfo], dim=1)
        subt_out = self.fc_layer01(subt_out)
        # endregion

        # region wild
        wild_out, _ = self.attention_layer02(wild_embedding, wild_embedding, wild_embedding)
        wild_out = self.middle_layer02(wild_out.view(-1, wild_out.shape[-1]))
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out, _ = self.attention_layer03(mutant_embedding, mutant_embedding, mutant_embedding)
        mutant_out = self.middle_layer03(mutant_out.view(-1, mutant_out.shape[-1]))
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = subt_out + (wild_out - mutant_out) / 2
        return output


class ESM3B_AttentionNet_BioInfo_Comb(nn.Module):
    def __init__(self,
                 trial=None):
        super(ESM3B_AttentionNet_BioInfo_Comb, self).__init__()

        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])

        self.attention_layer01 = MultiheadAttention(embed_dim=5120,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer01 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(5120),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer02 = MultiheadAttention(embed_dim=2560,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer02 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer03 = MultiheadAttention(embed_dim=2560,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer03 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, BioInfo):
        # region concating
        conc = torch.concat([wild_embedding, mutant_embedding], dim=2)
        conc_out, _ = self.attention_layer01(conc, conc, conc)
        conc_out = self.middle_layer01(conc_out.view(-1, conc_out.shape[-1]))
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), BioInfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out, _ = self.attention_layer02(wild_embedding, wild_embedding, wild_embedding)
        wild_out = self.middle_layer02(wild_out.view(-1, wild_out.shape[-1]))
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out, _ = self.attention_layer03(mutant_embedding, mutant_embedding, mutant_embedding)
        mutant_out = self.middle_layer03(mutant_out.view(-1, mutant_out.shape[-1]))
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


# endregion


# region ESM650M
class ESM650M_ConvNet_BioInfo_Sub(nn.Module):

    def __init__(self,
                 trial=None):
        super(ESM650M_ConvNet_BioInfo_Sub, self).__init__()
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        # region concating
        conc = torch.sub(wild_embedding, mutant_embedding)
        conc_out = self.conv_layer(conc)
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), bioinfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out = self.conv_layer(wild_embedding)
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out = self.conv_layer(mutant_embedding)
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


class ESM650M_ConvNet_BioInfo_Comb(nn.Module):

    def __init__(self,
                 trial=None):
        super(ESM650M_ConvNet_BioInfo_Comb, self).__init__()
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        # region concating
        conc = torch.concat([wild_embedding, mutant_embedding], dim=2)
        conc_out = self.conv_layer(conc)
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), bioinfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out = self.conv_layer(wild_embedding)
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out = self.conv_layer(mutant_embedding)
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


class ESM650M_AttentionNet_BioInfo_Sub(nn.Module):
    def __init__(self,
                 trial=None):
        super(ESM650M_AttentionNet_BioInfo_Sub, self).__init__()

        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])

        self.attention_layer01 = MultiheadAttention(embed_dim=1280,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer01 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1280),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer02 = MultiheadAttention(embed_dim=1280,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer02 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1280),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer03 = MultiheadAttention(embed_dim=1280,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer03 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1280),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, BioInfo):
        # region concating
        subt = torch.sub(wild_embedding, mutant_embedding)
        subt_out, _ = self.attention_layer01(subt, subt, subt)
        subt_out = self.middle_layer01(subt_out.view(-1, subt_out.shape[-1]))
        subt_out = torch.concat([subt_out.reshape(subt_out.shape[0], -1), BioInfo], dim=1)
        subt_out = self.fc_layer01(subt_out)
        # endregion

        # region wild
        wild_out, _ = self.attention_layer02(wild_embedding, wild_embedding, wild_embedding)
        wild_out = self.middle_layer02(wild_out.view(-1, wild_out.shape[-1]))
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out, _ = self.attention_layer03(mutant_embedding, mutant_embedding, mutant_embedding)
        mutant_out = self.middle_layer03(mutant_out.view(-1, mutant_out.shape[-1]))
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = subt_out + (wild_out - mutant_out) / 2
        return output


class ESM650M_AttentionNet_BioInfo_Comb(nn.Module):
    def __init__(self,
                 trial=None):
        super(ESM650M_AttentionNet_BioInfo_Comb, self).__init__()
        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])

        self.attention_layer01 = MultiheadAttention(embed_dim=2560,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer01 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2560),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer02 = MultiheadAttention(embed_dim=1280,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer02 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1280),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer03 = MultiheadAttention(embed_dim=1280,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer03 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1280),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, BioInfo):
        # region concating
        conc = torch.concat([wild_embedding, mutant_embedding], dim=2)
        conc_out, _ = self.attention_layer01(conc, conc, conc)
        conc_out = self.middle_layer01(conc_out.view(-1, conc_out.shape[-1]))
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), BioInfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out, _ = self.attention_layer02(wild_embedding, wild_embedding, wild_embedding)
        wild_out = self.middle_layer02(wild_out.view(-1, wild_out.shape[-1]))
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out, _ = self.attention_layer03(mutant_embedding, mutant_embedding, mutant_embedding)
        mutant_out = self.middle_layer03(mutant_out.view(-1, mutant_out.shape[-1]))
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


# endregion


# region ProtBert
class ProtBert_ConvNet_BioInfo_Sub(nn.Module):

    def __init__(self,
                 trial=None):
        super(ProtBert_ConvNet_BioInfo_Sub, self).__init__()
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        # region concating
        conc = torch.sub(wild_embedding, mutant_embedding)
        conc_out = self.conv_layer(conc)
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), bioinfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out = self.conv_layer(wild_embedding)
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out = self.conv_layer(mutant_embedding)
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


class ProtBert_ConvNet_BioInfo_Comb(nn.Module):

    def __init__(self,
                 trial=None):
        super(ProtBert_ConvNet_BioInfo_Comb, self).__init__()
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(1024), nn.PReLU(),
            torch.nn.LazyLinear(128),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        # region concating
        conc = torch.concat([wild_embedding, mutant_embedding], dim=2)
        conc_out = self.conv_layer(conc)
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), bioinfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out = self.conv_layer(wild_embedding)
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out = self.conv_layer(mutant_embedding)
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


class ProtBert_AttentionNet_BioInfo_Sub(nn.Module):
    def __init__(self,
                 trial=None):
        super(ProtBert_AttentionNet_BioInfo_Sub, self).__init__()

        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])

        self.attention_layer01 = MultiheadAttention(embed_dim=1024,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer01 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer02 = MultiheadAttention(embed_dim=1024,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer02 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer03 = MultiheadAttention(embed_dim=1024,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer03 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, BioInfo):
        # region concating
        subt = torch.sub(wild_embedding, mutant_embedding)
        subt_out, _ = self.attention_layer01(subt, subt, subt)
        subt_out = self.middle_layer01(subt_out.view(-1, subt_out.shape[-1]))
        subt_out = torch.concat([subt_out.reshape(subt_out.shape[0], -1), BioInfo], dim=1)
        subt_out = self.fc_layer01(subt_out)
        # endregion

        # region wild
        wild_out, _ = self.attention_layer02(wild_embedding, wild_embedding, wild_embedding)
        wild_out = self.middle_layer02(wild_out.view(-1, wild_out.shape[-1]))
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out, _ = self.attention_layer03(mutant_embedding, mutant_embedding, mutant_embedding)
        mutant_out = self.middle_layer03(mutant_out.view(-1, mutant_out.shape[-1]))
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = subt_out + (wild_out - mutant_out) / 2
        return output


class ProtBert_AttentionNet_BioInfo_Comb(nn.Module):
    def __init__(self,
                 trial=None):
        super(ProtBert_AttentionNet_BioInfo_Comb, self).__init__()

        dropout_rate_attention = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_bn = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear01 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear02 = trial.suggest_float("dropout", 0.2, 0.7)
        dropout_rate_linear03 = trial.suggest_float("dropout", 0.2, 0.7)
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        bias = trial.suggest_categorical("bias", [True, False])

        self.attention_layer01 = MultiheadAttention(embed_dim=2048,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer01 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2048),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer02 = MultiheadAttention(embed_dim=1024,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer02 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate_bn)
        )
        self.attention_layer03 = MultiheadAttention(embed_dim=1024,
                                                    num_heads=num_heads,
                                                    bias=bias,
                                                    dropout=dropout_rate_attention)
        self.middle_layer03 = nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate_bn)
        )

        self.fc_layer01 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear01),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer02 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear02),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )
        self.fc_layer03 = torch.nn.Sequential(
            torch.nn.LazyLinear(128), nn.PReLU(),
            nn.Dropout(dropout_rate_linear03),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, BioInfo):
        # region concating
        conc = torch.concat([wild_embedding, mutant_embedding], dim=2)
        conc_out, _ = self.attention_layer01(conc, conc, conc)
        conc_out = self.middle_layer01(conc_out.view(-1, conc_out.shape[-1]))
        conc_out = torch.concat([conc_out.reshape(conc_out.shape[0], -1), BioInfo], dim=1)
        conc_out = self.fc_layer01(conc_out)
        # endregion

        # region wild
        wild_out, _ = self.attention_layer02(wild_embedding, wild_embedding, wild_embedding)
        wild_out = self.middle_layer02(wild_out.view(-1, wild_out.shape[-1]))
        wild_out = self.fc_layer02(wild_out.reshape(wild_out.shape[0], -1))
        # endregion

        # region mutant
        mutant_out, _ = self.attention_layer03(mutant_embedding, mutant_embedding, mutant_embedding)
        mutant_out = self.middle_layer03(mutant_out.view(-1, mutant_out.shape[-1]))
        mutant_out = self.fc_layer03(mutant_out.reshape(mutant_out.shape[0], -1))
        # endregion

        output = conc_out + (wild_out - mutant_out) / 2
        return output


# endregion


if __name__ == "__main__":
    wild, mutant = torch.randn(64, 1, 1024).to(device="cuda", dtype=float_scale), torch.randn(64, 1, 1024).to(
        device="cuda", dtype=float_scale)
    bio = torch.randn(64, 642).cuda()
    model = ProtBert_ConvNet_BioInfo_Comb().to(device="cuda", dtype=float_scale)
    result = model(wild, mutant, bio)
