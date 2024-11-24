import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchmetrics
import warnings
import os
import logging
from datetime import datetime
from tqdm import tqdm
import sys

if sys.platform == "linux":
    os.chdir("/public/home/yyang/kjh/MutTm-pred")
    sys.path.append("/public/home/yyang/kjh/MutTm-pred")
else:
    os.chdir("D:/WorkPath/PycharmProjects/MutTm-pred")
    sys.path.append("D:/WorkPath/PycharmProjects/MutTm-pred")
from Dataset.Process4Dataset.DatasetCeator4PonDT import Dataset4MutTm
float_scale = torch.float64


# region General
def logger_init(logger, output_dir):
    """
    对日志对象self.logger进行初始化\n
    debug info warning error critical
    :param logger:
    :param output_dir: 日志保存目录
    :return: None
    """
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # output to file
    now = datetime.now()
    file_handler = logging.FileHandler(filename=f"./{output_dir}/{now.month}-{now.day}-{now.hour}-{now.minute}.log",
                                       mode="w",
                                       encoding="utf-8")
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)
    # output to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def embedding_model_getter(embedding_model_path: str,
                           embedding_model_name: str,
                           device):
    if embedding_model_name in {"ESM-2-150M", "ESM-2-650M", "ESM-2-3B"}:
        from transformers import EsmTokenizer, EsmModel
        embedding_tokenizer = EsmTokenizer.from_pretrained(embedding_model_path)
        embedding_model = EsmModel.from_pretrained(embedding_model_path).to(device)
    elif embedding_model_name in {"ProtT5-Half", }:
        from transformers import T5Tokenizer, T5EncoderModel
        embedding_tokenizer = T5Tokenizer.from_pretrained(embedding_model_path, do_lower_case=False)
        embedding_model = T5EncoderModel.from_pretrained(embedding_model_path).to(device)
    elif embedding_model_name in {"ProtBert", }:
        from transformers import BertTokenizer, BertModel
        embedding_tokenizer = BertTokenizer.from_pretrained(embedding_model_path, do_lower_case=False, legacy=True)
        embedding_model = BertModel.from_pretrained(embedding_model_path).to(device)
    elif embedding_model_name in {"CARP"}:
        from sequence_models.pretrained import load_model_and_alphabet
        embedding_model, embedding_tokenizer = load_model_and_alphabet('carp_640M')
    else:
        from transformers import AutoTokenizer, AutoModel
        try:
            embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
            embedding_model = AutoModel.from_pretrained(embedding_model_path).to(device)
        except Exception:
            raise ValueError(f"Can not load the model from {embedding_model_path}, please check in huggingface.co")

    return embedding_tokenizer, embedding_model


class PonMetrics:

    def __init__(self,
                 device: torch.device = torch.device("cpu")):
        self.metrics = {}
        self.KFoldSave = []
        self.folder = 1
        self.device = device
        self.metrics["MAE"] = torchmetrics.MeanAbsoluteError()  # 平均绝对误差
        self.metrics["PCC"] = torchmetrics.PearsonCorrCoef()    # 皮尔森相关系数
        self.metrics["R^2"] = torchmetrics.R2Score()            # R2系数
        self.metrics["MSE"] = torchmetrics.MeanSquaredError()   # 均方误差

    def record(self, output, target) -> None:
        for key in self.metrics.keys():
            self.metrics[key](output, target)

    def display(self) -> str:
        out = []
        for key, value in self.metrics.items():
            out.append(key + ": " + str(value.compute()))
        return "\n".join(out)

    def metric_getter(self) -> list:
        flow = ["MAE", "PCC", "R^2", "MSE"]
        return [self.metrics[key].compute() for key in flow]

    def clear(self) -> None:
        for key in self.metrics.keys():
            self.metrics[key].reset()

    def saveKFolderMetrics(self):
        self.KFoldSave.append(str(self.folder))
        self.KFoldSave.append(self.display())

    def showKFolderResult(self):
        return "\n".join(self.KFoldSave)

    def to(self, device, dtype):
        for key in self.metrics.keys():
            self.metrics[key] = self.metrics[key].to(device=device, dtype=dtype)
        return self

    def cuda(self):
        if torch.cuda.is_available():
            return self.to(torch.device("cuda"))
        else:
            warnings.warn("Cuda is not available. Switching to CPU. Please check your device.")
            return self.cpu()

    def cpu(self):
        return self.to(torch.device("cpu"))


class EarlyStopping:

    def __init__(self,
                 model_save_path,
                 patience,
                 verbose=False,
                 delta: float = 0.0):
        """
        :param model_save_path:模型保存路径
        :param patience:“耐心”值
        :param verbose:是否显示数据
        :param delta:差异值，用于衡量是否进行了“有效”训练
        """
        self.model_save_path = model_save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:  # 如果不存在最优值，即进行第一轮验证
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:  # 如果下降值小于预期，则认为开始进入无效区域
            self.counter += 1
            if self.counter >= self.patience:  # 如果超过一定次数，则停止训练
                self.early_stop = True
        else:  # 如果有明显下降则重写最小值，并保存模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # 同时重置计数器

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.model_save_path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

    def stop_or_not(self):
        return self.early_stop
# endregion


# region Use Embedding Model to fine tuning
class PonDataset4FineTuning(Dataset):
    def __init__(self,
                 wii: torch.Tensor | pd.DataFrame | list,
                 wam: torch.Tensor | pd.DataFrame | list,
                 mii: torch.Tensor | pd.DataFrame | list,
                 mam: torch.Tensor | pd.DataFrame | list,
                 bio_info: torch.Tensor | pd.DataFrame | list,
                 label: torch.Tensor | pd.DataFrame | list
                 ):
        # init
        self.wii = wii if isinstance(wii, torch.Tensor) else torch.stack(wii, dim=1)
        self.wam = wam if isinstance(wam, torch.Tensor) else torch.stack(wam, dim=1)
        self.mii = mii if isinstance(mii, torch.Tensor) else torch.stack(mii, dim=1)
        self.mam = mam if isinstance(mam, torch.Tensor) else torch.stack(mam, dim=1)
        self.bio_info = torch.Tensor(bio_info).to(dtype=torch.float32)
        self.label = torch.Tensor(label).to(dtype=torch.float32).unsqueeze(1)

    def subDataset(self, index):
        new = PonDataset4FineTuning(self.wii, self.wam, self.mii, self.mam, self.bio_info, self.label)
        new.setDataset(*self.__getitem__(index))
        return new

    def setDataset(self,
                   wii: torch.Tensor,
                   wam: torch.Tensor,
                   mii: torch.Tensor,
                   mam: torch.Tensor,
                   bio_info: torch.Tensor,
                   label: torch.Tensor):
        self.wii = wii
        self.wam = wam
        self.mii = mii
        self.mam = mam
        self.bio_info = bio_info
        self.label = label

    def __getitem__(self, mask) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        mii = self.mii[mask]
        mam = self.mam[mask]
        wii = self.wii[mask]
        wam = self.wam[mask]
        bioInfo = self.bio_info[mask]
        label = self.label[mask]
        return mii, mam, wii, wam, bioInfo, label

    def __len__(self) -> int:
        return len(self.label)
# endregion


# region Use EmbeddingModel as feature extractor
class PonDataset4FeatureExtraction(Dataset):
    def __init__(self,
                 wild_embedding: torch.Tensor | pd.DataFrame | list | np.ndarray,
                 mutant_embedding: torch.Tensor | pd.DataFrame | list | np.ndarray,
                 bio_info: torch.Tensor | pd.DataFrame | list | np.ndarray,
                 label: torch.Tensor | pd.DataFrame | list | np.ndarray
                 ):
        # init
        self.wild_embedding = wild_embedding.to(dtype=float_scale) \
            if isinstance(wild_embedding,torch.Tensor) \
            else torch.tensor(wild_embedding, dtype=float_scale)
        self.mutant_embedding = mutant_embedding.to(dtype=float_scale) \
            if isinstance(mutant_embedding, torch.Tensor) \
            else torch.tensor(mutant_embedding, dtype=float_scale)
        self.bio_info = bio_info.to(dtype=float_scale) \
            if isinstance(bio_info, torch.Tensor) \
            else torch.tensor(bio_info, dtype=float_scale)
        self.label = label.to(dtype=float_scale).unsqueeze(1) \
            if isinstance(label, torch.Tensor) \
            else torch.tensor(label, dtype=float_scale).unsqueeze(1)

    def subDataset(self, index):
        new = PonDataset4FeatureExtraction(self.wild_embedding, self.mutant_embedding, self.bio_info, self.label)
        new.setDataset(*self.__getitem__(index))
        return new

    def setDataset(self,
                   wild_embedding: torch.Tensor,
                   mutant_embedding: torch.Tensor,
                   bio_info: torch.Tensor,
                   label: torch.Tensor):
        self.wild_embedding = wild_embedding
        self.mutant_embedding = mutant_embedding
        self.bio_info = bio_info
        self.label = label

    def __getitem__(self, mask) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        wild_embedding = self.wild_embedding[mask]
        mutant_embedding = self.mutant_embedding[mask]
        bioInfo = self.bio_info[mask]
        label = self.label[mask]
        return wild_embedding, mutant_embedding, bioInfo, label

    def __len__(self) -> int:
        return len(self.label)


def dataset_creator4feature_extraction(df: Dataset4MutTm,
                                       embedding_model_name: str,
                                       model,
                                       tokenizer,
                                       device) -> (PonDataset4FeatureExtraction, PonDataset4FeatureExtraction):
    """
    Pre-trained models from Rostlab (beause of the special formet, every sequence need to transform)\n
    ProtT5, ProtBert can usethis function to build the testLoader.
    """
    # region 蛋白质嵌入(仅进行特征提取)
    with torch.no_grad():
        # select the basic information and tokenize
        processingBar = tqdm(total=4)
        if embedding_model_name in {"ESM-2-150M", "ESM-2-650M", "ESM-2-3B"}:
            train_wild_embedding = np.array([model(
                **(tokenizer(sequence, return_tensors="pt", padding=False).to(device))).last_hidden_state.mean(
                dim=1).cpu().numpy()
                                             for sequence in df.train_basic_set["WildSeq"].to_list()])
            processingBar.update(1)
            test_wild_embedding = np.array([model(**(tokenizer(sequence,
                                                               return_tensors="pt",
                                                               padding=False).to(device))).last_hidden_state.mean(
                dim=1).cpu().numpy()
                                            for sequence in df.test_basic_set["WildSeq"].to_list()])
            processingBar.update(1)
            train_mutant_embedding = np.array([model(**(tokenizer(sequence,
                                                                  return_tensors="pt",
                                                                  padding=False).to(device))).last_hidden_state.mean(
                dim=1).cpu().numpy()
                                               for sequence in df.train_basic_set["MutantSeq"].to_list()])
            processingBar.update(1)
            test_mutant_embedding = np.array([model(**(tokenizer(sequence,
                                                                 return_tensors="pt",
                                                                 padding=False).to(device))).last_hidden_state.mean(
                dim=1).cpu().numpy()
                                              for sequence in df.test_basic_set["MutantSeq"].to_list()])
            processingBar.update(1)
        elif embedding_model_name in {"ProtT5", "ProtBert"}:
            train_wild_embedding = np.array([model(**(tokenizer(" ".join(sequence),
                                                                return_tensors="pt",
                                                                padding=False).to(device))).last_hidden_state.mean(
                dim=1).cpu().numpy()
                                             for sequence in df.train_basic_set["WildSeq"].to_list()])
            processingBar.update(1)
            test_wild_embedding = np.array([model(**(tokenizer(" ".join(sequence),
                                                               return_tensors="pt",
                                                               padding=False).to(device))).last_hidden_state.mean(
                dim=1).cpu().numpy()
                                            for sequence in df.test_basic_set["WildSeq"].to_list()])
            processingBar.update(1)
            train_mutant_embedding = np.array([model(**(tokenizer(" ".join(sequence),
                                                                  return_tensors="pt",
                                                                  padding=False).to(device))).last_hidden_state.mean(
                dim=1).cpu().numpy()
                                               for sequence in df.train_basic_set["MutantSeq"].to_list()])
            processingBar.update(1)
            test_mutant_embedding = np.array([model(**(tokenizer(" ".join(sequence),
                                                                 return_tensors="pt",
                                                                 padding=False).to(device))).last_hidden_state.mean(
                dim=1).cpu().numpy()
                                              for sequence in df.test_basic_set["MutantSeq"].to_list()])
            processingBar.update(1)
        else:
            raise ValueError(
                f"Please check whether you forget to add this model name into DeepLearning/Util.py/embedding_model_getter()")
    processingBar.close()
    # endregion

    # region 生物信息提取
    train_bio_info = np.array(df.train_feature_set)
    test_bio_info = np.array(df.test_feature_set)
    # endregion

    # region DeltaTm标签
    train_labels = np.array(df.train_label_set)
    test_labels = np.array(df.test_label_set)
    # endregion

    # region 创建PonDataset4FeatureExtraction
    train_dataset = PonDataset4FeatureExtraction(wild_embedding=train_wild_embedding,
                                                 mutant_embedding=train_mutant_embedding,
                                                 bio_info=train_bio_info,
                                                 label=train_labels)
    test_dataset = PonDataset4FeatureExtraction(wild_embedding=test_wild_embedding,
                                                mutant_embedding=test_mutant_embedding,
                                                bio_info=test_bio_info,
                                                label=test_labels)
    return train_dataset, test_dataset
    # endregion


def training4feature_extraction(model,
                                trainLoader,
                                validateLoader,
                                loss_fn,
                                optimizer,
                                model_save_path,
                                device):
    model = model.to(device, dtype=float_scale)
    metrics = PonMetrics(device=device).to(device, dtype=float_scale)
    stoper = EarlyStopping(model_save_path,
                           30,
                           False,
                           0.01)

    for epoch in range(500):
        metrics.clear()

        # region Training
        train_loss = 0.0
        train_steps = 0
        model.train()
        for wild_embedding, mutant_embedding, bio, label in trainLoader:
            wild_embedding = wild_embedding.to(device, dtype=float_scale)
            mutant_embedding = mutant_embedding.to(device, dtype=float_scale)
            bio = bio.to(device, dtype=float_scale)
            label = label.to(device, dtype=float_scale)
            output = model(wild_embedding, mutant_embedding, bio)  # 预测模型
            running_loss = loss_fn(output, label)  # 损失函数并运行计算梯度

            optimizer.zero_grad()  # 优化器梯度清零
            running_loss.requires_grad_(True)  # 允许梯度
            running_loss.backward()  # 反向传播
            optimizer.step()  # 重置参数
            train_loss += running_loss.item()
            train_steps += 1

        # endregion

        # region Validation
        valid_loss = 0.0
        valid_steps = 0
        model.eval()
        with torch.no_grad():
            for wild_embedding, mutant_embedding, bio, label in validateLoader:
                # 载入GPU
                wild_embedding = wild_embedding.to(device, dtype=float_scale)
                mutant_embedding = mutant_embedding.to(device, dtype=float_scale)
                bio = bio.to(device, dtype=float_scale)
                label = label.to(device, dtype=float_scale)
                output = model(wild_embedding, mutant_embedding, bio)  # 预测模型
                metrics.record(output, label)  # 计算每一轮的指标
                running_loss = loss_fn(output, label)  # 损失函数并运行计算梯度
                valid_loss += running_loss.item()
                valid_steps += 1
        # endregion

        stoper(valid_loss, model)
        if stoper.stop_or_not():
            print(f"Training finished at Epoch {epoch}, with validation result for {[float(metric.cpu()) for metric in metrics.metric_getter()]}")
            break


def testing4feature_extraction(model,
                               testLoader,
                               model_params_path: str,
                               device) -> list:
    model.load_state_dict(torch.load(model_params_path))
    model = model.to(device, dtype=float_scale)
    metrics = PonMetrics(device=device).to(device, dtype=float_scale)

    # 开始测试
    model.eval()
    with torch.no_grad():
        for wild_embedding, mutant_embedding, bio, label in testLoader:
            wild_embedding = wild_embedding.to(device, dtype=float_scale)
            mutant_embedding = mutant_embedding.to(device, dtype=float_scale)
            bio = bio.to(device, dtype=float_scale)
            label = label.to(device, dtype=float_scale)
            output = model(wild_embedding, mutant_embedding, bio)
            metrics.record(output, label)

    return [float(metric.cpu()) for metric in metrics.metric_getter()]
# endregion


if __name__ == "__main__":
    from sequence_models.pretrained import load_model_and_alphabet
    model, collater = load_model_and_alphabet('carp_640M')
    seq = "MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQ"
    token = collater(seq)       # (n, max_len)
    embedding = model(token)        # (n, max_len, d_model)
# def rostlab_embedding_dataset_creator(df: Dataset4MutTm,
#                                       tokenizer) -> (PonDataset4FineTuning, PonDataset4FineTuning):
#     """
#     Pre-trained models from Rostlab (beause of the special formet, every sequence need to transform)\n
#     ProtT5, ProtBert can usethis function to build the testLoader.
#     :param df:
#     :param tokenizer:
#     :return:
#     """
#     # select the basic information and tokenize
#     train_wild_tokenized = tokenizer([" ".join(sequence) for sequence in df.train_basic_set["WildSeq"].to_list()], return_tensors="pt", padding=False)
#     test_wild_tokenized = tokenizer([" ".join(sequence) for sequence in df.test_basic_set["WildSeq"].to_list()], return_tensors="pt", padding=False)
#     train_mutant_tokenized = tokenizer([" ".join(sequence) for sequence in df.train_basic_set["MutantSeq"].to_list()], return_tensors="pt", padding=False)
#     test_mutant_tokenized = tokenizer([" ".join(sequence) for sequence in df.test_basic_set["MutantSeq"].to_list()], return_tensors="pt", padding=False)
#     train_bio_info = np.array(df.train_feature_set)
#     test_bio_info = np.array(df.test_feature_set)
#     train_labels = np.array(df.train_label_set)
#     test_labels = np.array(df.test_label_set)
#
#     # create the PonDataset4FineTuning
#     _train_dataset = PonDataset4FineTuning(wii=train_wild_tokenized["input_ids"],
#                                 wam=train_wild_tokenized["attention_mask"],
#                                 mii=train_mutant_tokenized["input_ids"],
#                                 mam=train_mutant_tokenized["attention_mask"],
#                                 bio_info=train_bio_info,
#                                 label=train_labels)
#     _test_dataset = PonDataset4FineTuning(wii=test_wild_tokenized["input_ids"],
#                                wam=test_wild_tokenized["attention_mask"],
#                                mii=test_mutant_tokenized["input_ids"],
#                                mam=test_mutant_tokenized["attention_mask"],
#                                bio_info=test_bio_info,
#                                label=test_labels)
#     return _train_dataset, _test_dataset
#
#
# def esm_embedding_dataset_creator(df: Dataset4MutTm,
#                                   tokenizer) -> (PonDataset4FineTuning, PonDataset4FineTuning):
#     # select the basic information and tokenize
#     train_wild_tokenized = tokenizer(df.train_basic_set["WildSeq"].to_list(), return_tensors="pt", padding=False)
#     test_wild_tokenized = tokenizer(df.test_basic_set["WildSeq"].to_list(), return_tensors="pt", padding=False)
#     train_mutant_tokenized = tokenizer(df.train_basic_set["MutantSeq"].to_list(), return_tensors="pt", padding=False)
#     test_mutant_tokenized = tokenizer(df.test_basic_set["MutantSeq"].to_list(), return_tensors="pt", padding=False)
#     train_bio_info = np.array(df.train_feature_set)
#     test_bio_info = np.array(df.test_feature_set)
#     train_labels = np.array(df.train_label_set)
#     test_labels = np.array(df.test_label_set)
#
#     # create the PonDataset4FineTuning
#     _train_dataset = PonDataset4FineTuning(wii=train_wild_tokenized["input_ids"],
#                                 wam=train_wild_tokenized["attention_mask"],
#                                 mii=train_mutant_tokenized["input_ids"],
#                                 mam=train_mutant_tokenized["attention_mask"],
#                                 bio_info=train_bio_info,
#                                 label=train_labels)
#     _test_dataset = PonDataset4FineTuning(wii=test_wild_tokenized["input_ids"],
#                                wam=test_wild_tokenized["attention_mask"],
#                                mii=test_mutant_tokenized["input_ids"],
#                                mam=test_mutant_tokenized["attention_mask"],
#                                bio_info=test_bio_info,
#                                label=test_labels)
#     return _train_dataset, _test_dataset


# class PonDataset4FeatureExtraction(Dataset):
#     def __init__(self,
#                  wild_embedding: torch.Tensor | pd.DataFrame | list | np.ndarray,
#                  mutant_embedding: torch.Tensor | pd.DataFrame | list | np.ndarray,
#                  bio_info: torch.Tensor | pd.DataFrame | list | np.ndarray,
#                  label: torch.Tensor | pd.DataFrame | list | np.ndarray
#                  ):
#         # init
#         self.wild_embedding = wild_embedding.to(dtype=torch.float32) if isinstance(wild_embedding,
#                                                                                    torch.Tensor) else torch.tensor(
#             wild_embedding, dtype=torch.float32)
#         self.mutant_embedding = mutant_embedding.to(dtype=torch.float32) if isinstance(mutant_embedding,
#                                                                                        torch.Tensor) else torch.tensor(
#             mutant_embedding, dtype=torch.float32)
#         self.bio_info = bio_info.to(dtype=torch.float32) if isinstance(bio_info, torch.Tensor) else torch.tensor(
#             bio_info, dtype=torch.float32)
#         self.label = label.to(dtype=torch.float32).unsqueeze(1) if isinstance(label, torch.Tensor) else torch.tensor(
#             label, dtype=torch.float32).unsqueeze(1)
#
#     def subDataset(self, index):
#         new = PonDataset4FeatureExtraction(self.wild_embedding, self.mutant_embedding, self.bio_info, self.label)
#         new.setDataset(*self.__getitem__(index))
#         return new
#
#     def setDataset(self,
#                    wild_embedding: torch.Tensor,
#                    mutant_embedding: torch.Tensor,
#                    bio_info: torch.Tensor,
#                    label: torch.Tensor):
#         self.wild_embedding = wild_embedding
#         self.mutant_embedding = mutant_embedding
#         self.bio_info = bio_info
#         self.label = label
#
#     def __getitem__(self, mask) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
#         wild_embedding = self.wild_embedding[mask]
#         mutant_embedding = self.mutant_embedding[mask]
#         bioInfo = self.bio_info[mask]
#         label = self.label[mask]
#         return wild_embedding, mutant_embedding, bioInfo, label
#
#     def __len__(self) -> int:
#         return len(self.label)
#

# def embedding_model_getter(embedding_model_path: str,
#                            embedding_model_name: str,
#                            device):
#     if embedding_model_name in {"ESM-2-150M", "ESM-2-650M", "ESM-2-3B"}:
#         from transformers import EsmTokenizer, EsmModel
#         embedding_tokenizer = EsmTokenizer.from_pretrained(embedding_model_path)
#         embedding_model = EsmModel.from_pretrained(embedding_model_path).to(device)
#     elif embedding_model_name in {"ProtT5-Half", }:
#         from transformers import T5Tokenizer, T5EncoderModel
#         embedding_tokenizer = T5Tokenizer.from_pretrained(embedding_model_path, do_lower_case=False)
#         embedding_model = T5EncoderModel.from_pretrained(embedding_model_path).to(device)
#     elif embedding_model_name in {"ProtBert", }:
#         from transformers import BertTokenizer, BertModel
#         embedding_tokenizer = BertTokenizer.from_pretrained(embedding_model_path, do_lower_case=False, legacy=True)
#         embedding_model = BertModel.from_pretrained(embedding_model_path)
#     else:
#         from transformers import AutoTokenizer, AutoModel
#         try:
#             embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
#             embedding_model = AutoModel.from_pretrained(embedding_model_path).to(device)
#         except Exception:
#             raise ValueError(f"Can not load the model from {embedding_model_path}, please check in huggingface.co")
#
#     return embedding_tokenizer, embedding_model
#
#
# def embedding_dataset_creator(df: Dataset4MutTm,
#                               embedding_model_name: str,
#                               tokenizer) -> (PonDataset4FineTuning, PonDataset4FineTuning):
#     """
#     Pre-trained models from Rostlab (beause of the special formet, every sequence need to transform)\n
#     ProtT5, ProtBert can usethis function to build the testLoader.
#     :param embedding_model_name:
#     :param df:
#     :param tokenizer:
#     :return:
#     """
#     # select the basic information and tokenize
#     if embedding_model_name in {"ESM-2-150M", "ESM-2-650M", "ESM-2-3B"}:
#         train_wild_tokenized = tokenizer(df.train_basic_set["WildSeq"].to_list(), return_tensors="pt", padding=False)
#         test_wild_tokenized = tokenizer(df.test_basic_set["WildSeq"].to_list(), return_tensors="pt", padding=False)
#         train_mutant_tokenized = tokenizer(df.train_basic_set["MutantSeq"].to_list(), return_tensors="pt",
#                                            padding=False)
#         test_mutant_tokenized = tokenizer(df.test_basic_set["MutantSeq"].to_list(), return_tensors="pt", padding=False)
#     elif embedding_model_name in {"ProtT5", "ProtBert"}:
#         train_wild_tokenized = tokenizer([" ".join(sequence) for sequence in df.train_basic_set["WildSeq"].to_list()],
#                                          return_tensors="pt", padding=False)
#         test_wild_tokenized = tokenizer([" ".join(sequence) for sequence in df.test_basic_set["WildSeq"].to_list()],
#                                         return_tensors="pt", padding=False)
#         train_mutant_tokenized = tokenizer(
#             [" ".join(sequence) for sequence in df.train_basic_set["MutantSeq"].to_list()],
#             return_tensors="pt", padding=False)
#         test_mutant_tokenized = tokenizer([" ".join(sequence) for sequence in df.test_basic_set["MutantSeq"].to_list()],
#                                           return_tensors="pt", padding=False)
#     else:
#         raise ValueError(
#             f"Please check whether you forget to add this model name into DeepLearning/Util.py/embedding_model_getter()")
#
#     train_bio_info = np.array(df.train_feature_set)
#     test_bio_info = np.array(df.test_feature_set)
#     train_labels = np.array(df.train_label_set)
#     test_labels = np.array(df.test_label_set)
#
#     # create the PonDataset4FineTuning
#     _train_dataset = PonDataset4FineTuning(wii=train_wild_tokenized["input_ids"],
#                                 wam=train_wild_tokenized["attention_mask"],
#                                 mii=train_mutant_tokenized["input_ids"],
#                                 mam=train_mutant_tokenized["attention_mask"],
#                                 bio_info=train_bio_info,
#                                 label=train_labels)
#     _test_dataset = PonDataset4FineTuning(wii=test_wild_tokenized["input_ids"],
#                                wam=test_wild_tokenized["attention_mask"],
#                                mii=test_mutant_tokenized["input_ids"],
#                                mam=test_mutant_tokenized["attention_mask"],
#                                bio_info=test_bio_info,
#                                label=test_labels)
#     return _train_dataset, _test_dataset
#
#
# def embedding_dataset_creator_full_length(df: Dataset4MutTm,
#                                           embedding_model_name: str,
#                                           model,
#                                           tokenizer,
#                                           device) -> (PonDataset4FineTuning, PonDataset4FineTuning):
#     """
#     Pre-trained models from Rostlab (beause of the special formet, every sequence need to transform)\n
#     ProtT5, ProtBert can usethis function to build the testLoader.
#     :param model:
#     :param device:
#     :param embedding_model:
#     :param embedding_model_name:
#     :param df:
#     :param tokenizer:
#     :return:
#     """
#     with torch.no_grad():
#         # select the basic information and tokenize
#         processingBar = tqdm.tqdm(total=4)
#         if embedding_model_name in {"ESM-2-150M", "ESM-2-650M", "ESM-2-3B"}:
#             train_wild_embedding = np.array([model(**(tokenizer(sequence, return_tensors="pt", padding=False).to(device))).last_hidden_state.mean(dim=1).cpu().numpy()
#                                              for sequence in df.train_basic_set["WildSeq"].to_list()])
#             processingBar.update(1)
#             test_wild_embedding = np.array([model(**(tokenizer(sequence,
#                                                                return_tensors="pt",
#                                                                padding=False).to(device))).last_hidden_state.mean(dim=1).cpu().numpy()
#                                             for sequence in df.test_basic_set["WildSeq"].to_list()])
#             processingBar.update(1)
#             train_mutant_embedding = np.array([model(**(tokenizer(sequence,
#                                                                   return_tensors="pt",
#                                                                   padding=False).to(device))).last_hidden_state.mean(dim=1).cpu().numpy()
#                                                for sequence in df.train_basic_set["MutantSeq"].to_list()])
#             processingBar.update(1)
#             test_mutant_embedding = np.array([model(**(tokenizer(sequence,
#                                                                  return_tensors="pt",
#                                                                  padding=False).to(device))).last_hidden_state.mean(dim=1).cpu().numpy()
#                                               for sequence in df.test_basic_set["MutantSeq"].to_list()])
#             processingBar.update(1)
#         elif embedding_model_name in {"ProtT5", "ProtBert"}:
#             train_wild_embedding = np.array([model(**(tokenizer(" ".join(sequence),
#                                                                 return_tensors="pt",
#                                                                 padding=False).to(device))).last_hidden_state.mean(dim=1).cpu().numpy()
#                                              for sequence in df.train_basic_set["WildSeq"].to_list()])
#             processingBar.update(1)
#             test_wild_embedding = np.array([model(**(tokenizer(" ".join(sequence),
#                                                                return_tensors="pt",
#                                                                padding=False).to(device))).last_hidden_state.mean(dim=1).cpu().numpy()
#                                             for sequence in df.test_basic_set["WildSeq"].to_list()])
#             processingBar.update(1)
#             train_mutant_embedding = np.array([model(**(tokenizer(" ".join(sequence),
#                                                                   return_tensors="pt",
#                                                                   padding=False).to(device))).last_hidden_state.mean(dim=1).cpu().numpy()
#                                                for sequence in df.train_basic_set["MutantSeq"].to_list()])
#             processingBar.update(1)
#             test_mutant_embedding = np.array([model(**(tokenizer(" ".join(sequence),
#                                                                  return_tensors="pt",
#                                                                  padding=False).to(device))).last_hidden_state.mean(dim=1).cpu().numpy()
#                                               for sequence in df.test_basic_set["MutantSeq"].to_list()])
#             processingBar.update(1)
#         else:
#             raise ValueError(
#                 f"Please check whether you forget to add this model name into DeepLearning/Util.py/embedding_model_getter()")
#     processingBar.close()
#
#     train_bio_info = np.array(df.train_feature_set)
#     test_bio_info = np.array(df.test_feature_set)
#     train_labels = np.array(df.train_label_set)
#     test_labels = np.array(df.test_label_set)
#
#     # create the PonDataset4FineTuning
#     _train_dataset = PonDataset4Conv(wild_embedding=train_wild_embedding,
#                                      mutant_embedding=train_mutant_embedding,
#                                      bio_info=train_bio_info,
#                                      label=train_labels)
#     _test_dataset = PonDataset4Conv(wild_embedding=test_wild_embedding,
#                                     mutant_embedding=test_mutant_embedding,
#                                     bio_info=test_bio_info,
#                                     label=test_labels)
#     return _train_dataset, _test_dataset
