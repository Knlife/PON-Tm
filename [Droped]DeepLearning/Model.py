from torch.nn import init
from torch import nn
from torch.nn import functional as F
import math
import torch


# region Based on FeedAttention
class FeedAttention(nn.Module):
    """
    InitializationDataset:https://github.com/WenYanger/Contextual-Attention/blob/master/Attention_Pytorch.py
    """

    def __init__(self, input_shape):
        super(FeedAttention, self).__init__()

        self.max_len = input_shape[1]
        self.emb_size = input_shape[2]

        # Change double to float
        self.weight = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.bias = nn.Parameter(torch.Tensor(self.max_len, 1))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return 'max_len={}, emb_size={}'.format(
            self.max_len, self.emb_size
        )

    def forward(self, inp, mask=None):
        """

        :param inp: sr should be [batch_size, time_step, emb_size]
        :param mask: mask should be [batch_size, time_step, 1]
        :return:
        """
        W_bs = self.weight.unsqueeze(0).repeat(inp.size()[0], 1,
                                               1).float()  # Copy the FeedAttention Matrix for batch_size times
        scores = torch.bmm(inp, W_bs)  # Dot product between input and attention matrix
        scores = torch.tanh(scores)

        if mask is not None:
            mask = mask.long()
            scores = scores.masked_fill(mask == 0, -1e9)

        a_ = F.softmax(scores.squeeze(-1), dim=-1)
        a = a_.unsqueeze(-1).repeat(1, 1, inp.size()[2])

        weighted_input = inp * a

        output = torch.sum(weighted_input, dim=1)

        return output


class PontBert_FeedAttention(nn.Module):
    """
    ProtBert 嵌入后的输出大小为[context_length, 1024]
    """

    def __init__(self,
                 length,
                 embedding_model,
                 embedding_model_name: str,
                 fine_tuning: bool = True):
        super(PontBert_FeedAttention, self).__init__()

        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.fine_tuning = fine_tuning

        self.feature_extraction_module = self.feed_attention_module = nn.Sequential(
            FeedAttention([None, length, 1024]),
            nn.BatchNorm1d(1024, dtype=torch.float32),
            nn.Dropout(0.4),
        )

        self.fussion_module4protbert = nn.Sequential(
            nn.Linear(1024, 512, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(512, 256, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(256, 128, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(128, 64, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(64, 16, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(16, 1, dtype=torch.float32)
        )

    def forward(self, wild_ids, wild_mask, mutant_ids, mutant_mask, bioinfo):
        # region 预训练特征模块-可选择是否进行微调
        if self.fine_tuning:
            wild_output = self.embedding_model(input_ids=wild_ids, attention_mask=wild_mask).last_hidden_state
            mutant_output = self.embedding_model(input_ids=mutant_ids, attention_mask=mutant_mask).last_hidden_state
        else:
            with torch.no_grad():
                wild_output = self.embedding_model(input_ids=wild_ids, attention_mask=wild_mask).last_hidden_state
                mutant_output = self.embedding_model(input_ids=mutant_ids, attention_mask=mutant_mask).last_hidden_state
        # endregion

        # region Feature Extraction
        wild_output = self.feature_extraction_module(wild_output)
        mutant_output = self.feature_extraction_module(mutant_output)
        # endregion

        # region Fussion
        wild_output = self.fussion_module4protbert(wild_output)
        mutant_output = self.fussion_module4protbert(mutant_output)
        out = torch.sub(mutant_output, wild_output)
        # endregion

        return out


class PonsT5_FeedAttention(nn.Module):
    """
    ProtBert 嵌入后的输出大小为[context_length, 1024]
    """

    def __init__(self,
                 length,
                 embedding_model,
                 embedding_model_name: str,
                 fine_tuning: bool = True):
        super(PonsT5_FeedAttention, self).__init__()

        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.fine_tuning = fine_tuning

        self.feature_extraction_module = self.feed_attention_module = nn.Sequential(
            FeedAttention([None, length, 1024]),
            nn.BatchNorm1d(1024, dtype=torch.float32),
            nn.Dropout(0.4),
        )

        self.fussion_module4protbert = nn.Sequential(
            nn.Linear(1024, 512, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(512, 256, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(256, 128, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(128, 64, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(64, 16, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(16, 1, dtype=torch.float32)
        )

    def forward(self, wild_ids, wild_mask, mutant_ids, mutant_mask, bioinfo):
        # region 预训练特征模块-可选择是否进行微调
        with torch.no_grad():
            wild_output = self.embedding_model(input_ids=wild_ids, attention_mask=wild_mask).last_hidden_state
            mutant_output = self.embedding_model(input_ids=mutant_ids, attention_mask=mutant_mask).last_hidden_state
        # endregion

        # region Feature Extraction
        wild_output = self.feature_extraction_module(wild_output)
        mutant_output = self.feature_extraction_module(mutant_output)
        # endregion

        # region Fussion
        wild_output = self.fussion_module4protbert(wild_output)
        mutant_output = self.fussion_module4protbert(mutant_output)
        out = torch.sub(mutant_output, wild_output)
        # endregion

        return out


class PonT5_Half_FeedAttention(nn.Module):
    """
    ProtBert 嵌入后的输出大小为[context_length, 1024]
    """

    def __init__(self,
                 length,
                 embedding_model,
                 embedding_model_name: str,
                 fine_tuning: bool = True):
        super(PonT5_Half_FeedAttention, self).__init__()

        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.fine_tuning = fine_tuning

        self.feature_extraction_module = self.feed_attention_module = nn.Sequential(
            FeedAttention([None, length, 1024]),
            nn.BatchNorm1d(1024, dtype=torch.float32),
            nn.Dropout(0.4),
        )

        self.fussion_module4protbert = nn.Sequential(
            nn.Linear(1024, 512, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(512, 256, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(256, 128, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(128, 64, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(64, 16, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(16, 1, dtype=torch.float32)
        )

    def forward(self, wild_ids, wild_mask, mutant_ids, mutant_mask, bioinfo):
        # region 预训练特征模块-可选择是否进行微调
        if self.fine_tuning:
            wild_output = self.embedding_model(input_ids=wild_ids, attention_mask=wild_mask).last_hidden_state
            mutant_output = self.embedding_model(input_ids=mutant_ids, attention_mask=mutant_mask).last_hidden_state
        else:
            with torch.no_grad():
                wild_output = self.embedding_model(input_ids=wild_ids, attention_mask=wild_mask).last_hidden_state
                mutant_output = self.embedding_model(input_ids=mutant_ids, attention_mask=mutant_mask).last_hidden_state
        # endregion

        # region Feature Extraction
        wild_output = self.feature_extraction_module(wild_output)
        mutant_output = self.feature_extraction_module(mutant_output)
        # endregion

        # region Fussion
        wild_output = self.fussion_module4protbert(wild_output)
        mutant_output = self.fussion_module4protbert(mutant_output)
        out = torch.sub(mutant_output, wild_output)
        # endregion

        return out


# endregion


# region Embedding + Conv
class ESM3B_ConvNet_Sub(nn.Module):

    def __init__(self):
        super(ESM3B_ConvNet_Sub, self).__init__()

        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),

            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(2560 * 32, 128),
            nn.Dropout(0.55),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding, bioinfo):
        output = torch.sub(wild_embedding, mutant_embedding)
        output = self.conv_layer(output)
        output = self.fc_layer(output.reshape(output.shape[0], -1))
        return output


class ESM3B_ConvNet_Comb(nn.Module):

    def __init__(self,
                 length: int,
                 embedding_model,
                 embedding_model_name: str,
                 fine_tuning: bool = True
                 ):
        super(ESM3B_ConvNet_Comb, self).__init__()

        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(5120 * 32, 1024),
            torch.nn.Linear(1024, 128),
            nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output = self.conv_layer(output)
        output = self.fc_layer(output.reshape(output.shape[0], -1))
        return output


class ESM650M_ConvNet_Sub(nn.Module):

    def __init__(self,
                 ):
        super(ESM650M_ConvNet_Sub, self).__init__()
        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(5120 * 32, 1024),
            torch.nn.Linear(1024, 128),
            nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output = self.conv_layer(output)
        output = self.fc_layer(output.reshape(output.shape[0], -1))
        return output


class ESM650M_ConvNet_Comb(nn.Module):

    def __init__(self,
                 length: int,
                 embedding_model,
                 embedding_model_name: str,
                 fine_tuning: bool = True
                 ):
        super(ESM650M_ConvNet_Comb, self).__init__()

        self.conv_layer = nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(5120 * 32, 1024),
            torch.nn.Linear(1024, 128),
            nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
        )

    def forward(self, wild_embedding, mutant_embedding):
        output = torch.concat([wild_embedding, mutant_embedding], dim=2)
        output = self.conv_layer(output)
        output = self.fc_layer(output.reshape(output.shape[0], -1))
        return output


# endregion


# region Droped Network
class FeatureExtration4ProtBert(nn.Module):
    def __init__(self, length):
        super(FeatureExtration4ProtBert, self).__init__()

        self.feed_attention_module = nn.Sequential(
            FeedAttention([None, length, 1024]),
            nn.BatchNorm1d(1024, dtype=torch.float32),
            nn.Dropout(0.3),
        )

    def forward(self, embedding):
        output = self.feed_attention_module(embedding)
        return output


class FeatureExtration4ESM150M(nn.Module):
    def __init__(self, length):
        super(FeatureExtration4ESM150M, self).__init__()

        self.feed_attention_module = nn.Sequential(
            FeedAttention([None, length, 640]),
            nn.BatchNorm1d(640, dtype=torch.float32),
            nn.Dropout(0.3),
        )

    def forward(self, embedding):
        output = self.feed_attention_module(embedding)
        return output


class PonDT(nn.Module):

    def __init__(self,
                 length,
                 embedding_model,
                 embedding_model_name: str,
                 fine_tuning: bool = True):
        super(PonDT, self).__init__()

        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.fine_tuning = fine_tuning

        # region 特征提取模块初始化
        if self.embedding_model_name == "ProtBert":
            self.feature_extraction_module = FeatureExtration4ProtBert(length)
        elif self.embedding_model_name == 'ESM-2-150M':
            self.feature_extraction_module = FeatureExtration4ESM150M(length)
        else:
            raise ValueError(
                f"Please check whether you forget to add this model name into DeepLearning/Model.py/PonDT.__init__().")
        # endregion

        # region 回归模块初始化
        self.Fussion_Module = nn.Sequential(
            nn.Linear(1280, 512, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(512, 256, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(256, 128, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(128, 32, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(32, 1, dtype=torch.float32)
        )
        self.fussion_module4protbert = nn.Sequential(
            nn.Linear(1024, 512, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(512, 256, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(256, 128, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(128, 32, dtype=torch.float32), nn.ReLU(inplace=True),
            nn.Linear(32, 1, dtype=torch.float32)
        )
        # endregion

    def forward(self, wild_ids, wild_mask, mutant_ids, mutant_mask, bioinfo):
        # region 预训练特征模块-可选择是否进行微调
        if self.fine_tuning:
            wild_output = self.embedding_model(input_ids=wild_ids, attention_mask=wild_mask).last_hidden_state
            mutant_output = self.embedding_model(input_ids=mutant_ids, attention_mask=mutant_mask).last_hidden_state
        else:
            with torch.no_grad():
                wild_output = self.embedding_model(input_ids=wild_ids, attention_mask=wild_mask).last_hidden_state
                mutant_output = self.embedding_model(input_ids=mutant_ids, attention_mask=mutant_mask).last_hidden_state
        # endregion

        # region 特征提取模块
        if self.embedding_model_name == "ProtBert":
            wild_output = self.feature_extraction_module(wild_output)
            mutant_output = self.feature_extraction_module(mutant_output)
        elif self.embedding_model_name == "ESM-2-150M":
            wild_output = self.feature_extraction_module(wild_output)
            mutant_output = self.feature_extraction_module(mutant_output)
        else:
            raise RuntimeError("Please specify the model for the new embedding model, ")
        # endregion

        # region 回归模块
        if self.embedding_model_name == "ProtBert":
            wild_output = self.fussion_module4protbert(wild_output)
            mutant_output = self.fussion_module4protbert(mutant_output)
        elif self.embedding_model_name == "ESM-2-150M":
            wild_output = self.fussion_module4esm150M(wild_output)
            mutant_output = self.fussion_module4esm150M(mutant_output)
        else:
            raise RuntimeError("Please specify the model for the new embedding model, ")
        # endregion

        out = torch.sub(mutant_output, wild_output)
        # output
        return out


# endregion


if __name__ == "__main__":
    inp = torch.randn([32, 1, 2560])
    conv1 = torch.nn.Conv1d(1, 8, 3, padding=1)
    conv2 = torch.nn.Conv1d(8, 32, 3, padding=1)
    output01 = conv1(inp)
    output02 = conv2(output01)

    fc_layer = torch.nn.Sequential(
        torch.nn.Linear(2560 * 32, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.Linear(128, 1),
    )
    output03 = output02.view(output02.size()[0], -1)
    output04 = fc_layer(output03)
