import math
import torch
import pandas as pd
import numpy as np
import os
import sys

from torch import device as torch_device, cuda

# feature extractors
if sys.platform == "linux":
    os.chdir("/public/home/yyang/kjh/MutTm-pred")
    sys.path.append("/public/home/yyang/kjh/MutTm-pred")
else:
    os.chdir("D:/WorkPath/PycharmProjects/MutTm-pred")
    sys.path.append("D:/WorkPath/PycharmProjects/MutTm-pred")
    from Dataset.Process4Dataset.feature_extractor import (uid2seq, aaindexGetter, neighorGetter, groupGetter,
                                                           siftGetter, pssmGetter, protrGetter, paramGetter, GOGetter,
                                                           embeddingGetter, hydropGetter, swisspssmGetter)

from Dataset.Process4Dataset.feature_extractor import aaindexGetter, embeddingGetter
from Dataset.Process4Dataset.initializor4PonDT import (file2df, wash4MutTm, duplicate4MutTm, check4mult,
                                                       delete4Dup, sequence_cutter, unipdb_getter)

# aminoacid
a_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
a_set = set(a_list)
aa_list = [i + j for i in a_list for j in a_list]


def necessity4MutTm(df: pd.DataFrame,
                    refelction_cache: str,
                    context_length: int) -> pd.DataFrame:
    # region Acquire (ProteinSeq, Fasta) from Uniprot
    # Acquire (ProteinSeq, Fasta) from Uniprot, if generate before, use cache
    print("-获取[序列]信息", end=".....")
    if os.path.exists(f"{refelction_cache}/seq.npy"):  # 如果已经存在之前获取过的序列信息则无需再次获取
        print("该数据集已经经过处理，直接使用缓存文件")
        seq_reflection = np.load(f"{refelction_cache}/seq.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
        seq_reflection = uid2seq.mult_uid(df["UniProt_ID"].drop_duplicates(keep="first"))
        np.save(f"{refelction_cache}/seq.npy", seq_reflection)

    deleteIndexs = []
    # delete the illegal proteins that cannot be found in Uniprot
    ss = {"A0A7H0P088", "P02640 ", "Q9S446 ", "P04156 ", "P17350", "Q9R782", "A0A410ZNC6", "A0A140NCX4"}
    for key, value in seq_reflection.items():
        if value[0] == "" or value[0] is None:
            ss.add(key)
    for rowIndex, row in df.iterrows():
        if row["UniProt_ID"] in ss:
            deleteIndexs.append(rowIndex)
    print(f"-删除了{len(deleteIndexs)}个UniProtID无法获取序列的数据")
    df = df.drop(deleteIndexs).reset_index(drop=True)
    deleteIndexs.clear()

    df["ProteinSeq"] = df["UniProt_ID"].apply(lambda ID: seq_reflection[ID][0])
    df["Fasta"] = df["UniProt_ID"].apply(lambda ID: seq_reflection[ID][1])

    # delete the protein of illegal length
    illegal_length_proteins = []
    for rowIndex, row in df.iterrows():
        if len(row["ProteinSeq"]) >= 5000 or len(row["ProteinSeq"]) <= context_length:
            illegal_length_proteins.append(rowIndex)
    df = df.drop(illegal_length_proteins).reset_index(drop=True)
    print(f"-删除了{len(illegal_length_proteins)}条非法长度的数据，当前蛋白质长度被限制在({context_length}, {5000})")
    deleteIndexs.clear()

    # endregion

    # region Acquire (Length, AminoAcidFrom, MutationIndex, AminoAcidTo, Positive) from (ProteinSeq, Mutation, ΔTm)
    df["Length"] = df["ProteinSeq"].apply(lambda x: len(x))
    df["AminoAcidFrom"] = df["Mutation"].apply(lambda mutation: mutation[0])
    df["MutationIndex"] = df["Mutation"].apply(lambda mutation: mutation[1:-1])
    df["AminoAcidTo"] = df["Mutation"].apply(lambda mutation: mutation[-1])
    df["Positive"] = df["ΔTm"].apply(lambda delta: 1 if float(delta) > 0 else 0)
    # endregion

    # region Delete proteins containing illegal aminoacid or having wrong mutation index
    for rowindex, row in df.iterrows():
        if row["AminoAcidFrom"] not in a_set or row["AminoAcidTo"] not in a_set or int(row["MutationIndex"]) > int(
                row["Length"]):
            deleteIndexs.append(rowindex)
            continue
        if int(row["MutationIndex"]) >= row["Length"]:
            deleteIndexs.append(rowindex)
            continue
        # if the position dont match the aminoacid, drop
        if row["ProteinSeq"][int(row["MutationIndex"]) - 1] != row["AminoAcidFrom"]:
            deleteIndexs.append(rowindex)
    print(f"-删除条{len(deleteIndexs)}个突变位点对应错误的数据")
    df = df.drop(index=deleteIndexs).reset_index(drop=True)
    deleteIndexs.clear()
    # endregion

    # get the (WildSeq, MutantSeq, WildCutSeq, MutantCutSeq) from (ProteinSeq, MutationIndex, AminoAcidTo)
    if not context_length:
        print("-不针对突变位点上下文进行截取")
    df["WildSeq"], df["MutantSeq"], df["WildCutSeq"], df["MutantCutSeq"] = \
        sequence_cutter(df[["ProteinSeq", "MutationIndex", "AminoAcidTo"]], context_length)
    return df


def neighbor4MutTm(df: pd.DataFrame, reflection_cache: str) -> pd.DataFrame:
    print("-获取[neighbor特征]", end=".....")
    if os.path.exists(f"{reflection_cache}/neighbor.npy") or sys.platform == "linux":
        print("该数据集已经经过处理，直接使用缓存文件")
        neighbor_reflection = np.load(f"{reflection_cache}/neighbor.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
        neighbor_reflection = neighorGetter.neighbor_getter(df)
        np.save(f"{reflection_cache}/neighbor.npy", neighbor_reflection)

    result = pd.DataFrame(columns=["neighbor"], index=df.index, dtype=object)
    for rowIndex, mixture in df[["UniProt_ID", "MutationIndex"]].iterrows():
        result.at[rowIndex, "neighbor"] = neighbor_reflection[
            mixture["UniProt_ID"] + "-" + str(int(mixture["MutationIndex"]))]
    result = pd.DataFrame(result["neighbor"].tolist(),
                          columns=[f"neighbor{_}" for _ in range(1, len(result.iloc[0, 0]) + 1)])
    return result


def aaindex4MutTm(df: pd.DataFrame, reflection_cache: str) -> pd.DataFrame:
    print("-获取[aaindex特征]", end=".....")
    if os.path.exists(f"{reflection_cache}/aaindex.npy"):
        print("该数据集已经经过处理，直接使用缓存文件")
        aaindex_reflection = np.load(f"{reflection_cache}/aaindex.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
    aaindex_reflection = aaindexGetter.aaindex_got_from_file()
    np.save(f"{reflection_cache}/aaindex.npy", aaindex_reflection)

    result = pd.DataFrame(columns=["aaindex"], index=df.index, dtype=object)
    for rowIndex, mixture in df[["AminoAcidFrom", "AminoAcidTo"]].iterrows():
        result.at[rowIndex, "aaindex"] = aaindex_reflection[mixture["AminoAcidFrom"] + mixture["AminoAcidTo"]]
    result = pd.DataFrame(result["aaindex"].tolist(),
                          columns=[f"aaindex{_}" for _ in range(1, len(result.iloc[0, 0]) + 1)])
    return result


def group4MutTm(df: pd.DataFrame, reflection_cache: str) -> pd.DataFrame:
    print("-获取[group特征]", end=".....")
    if os.path.exists(f"{reflection_cache}/group.npy") or sys.platform == "linux":
        print("该数据集已经经过处理，直接使用缓存文件")
        group_reflection = np.load(f"{reflection_cache}/group.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
        group_reflection = groupGetter.mult_getter(df)
        np.save(f"{reflection_cache}/group.npy", group_reflection)

    result = pd.DataFrame(columns=["group"], index=df.index, dtype=object)
    for rowIndex, mixture in df[["AminoAcidFrom", "AminoAcidTo"]].iterrows():
        result.at[rowIndex, "group"] = group_reflection["".join(map(str, mixture))]
    result = pd.DataFrame(result["group"].tolist(),
                          columns=[f"group{_}" for _ in range(1, len(result.iloc[0, 0]) + 1)])
    return result


def sift4MutTm(df: pd.DataFrame,
               reflection_cache: str,
               dataset_version: str) -> pd.DataFrame:
    print("-获取[sift4g]特征", end=".....")
    if os.path.exists(f"{reflection_cache}/sift4g.npy") or sys.platform == "linux":
        print("该数据集已经经过处理，直接使用缓存文件")
        sift4g_reflection = np.load(f"{reflection_cache}/sift4g.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
        sift4g_reflection = siftGetter.mult_getter(df, dataset_version=dataset_version)
        np.save(f"{reflection_cache}/sift4g.npy", sift4g_reflection)

    result = pd.DataFrame(columns=["sift4g_scores"], index=df.index, dtype=object)
    for rowIndex, mixture in df[["UniProt_ID", "Mutation"]].iterrows():
        result.at[rowIndex, "sift4g_scores"] = list(
            sift4g_reflection["-".join(map(str, [mixture["UniProt_ID"], mixture["Mutation"]]))])
    result = pd.DataFrame(result["sift4g_scores"].tolist(),
                          columns=[f"sift4g_scores{_}" for _ in range(1, len(result.iloc[0, 0]) + 1)])
    return result


def pssm4MutTm(df: pd.DataFrame, reflection_cache: str) -> pd.DataFrame:
    print("-获取[pssm]特征", end=".....")
    if os.path.exists(f"{reflection_cache}/rpm.npy") or sys.platform == "linux":
        print("该数据集已经经过处理，直接使用缓存文件")
        rpm_reflection = np.load(f"{reflection_cache}/rpm.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
        rpm_reflection = pssmGetter.PssmGotFromUrl(df, "RPM_PSSM")
        np.save(f"{reflection_cache}/rpm.npy", rpm_reflection)

    result = df["UniProt_ID"].apply(lambda ID: rpm_reflection[ID])
    result = pd.DataFrame(result.tolist(), columns=[f"pssm{_}" for _ in range(1, result.iloc[0].shape[0] + 1)])
    return result


def swisspssm4MutTm(df: pd.DataFrame, reflection_cache: str) -> pd.DataFrame:
    print("-获取[swiss-pssm]特征", end=".....")

    if os.path.exists(f"{reflection_cache}/rpm-swiss.npy"):
        print("该数据集已经经过处理，直接使用缓存文件")
        rpm_reflection = np.load(f"{reflection_cache}/rpm-swiss.npy", allow_pickle=True).item()
    else:
        rpm_reflection = swisspssmGetter.PssmGotFromUrl(df, "RPM_PSSM")
        np.save(f"{reflection_cache}/rpm-swiss.npy", rpm_reflection)

    result = df["UniProt_ID"].apply(lambda ID: rpm_reflection[ID])
    result = pd.DataFrame(result.tolist(), columns=[f"pssm{_}" for _ in range(1, result.iloc[0].shape[0] + 1)])
    return result


def protr4MutTm(df: pd.DataFrame, reflection_cache: str) -> pd.DataFrame:
    print("-获取[protr]特征", end=".....")
    if os.path.exists(f"{reflection_cache}/protr.npy") or sys.platform == "linux":
        print("该数据集已经经过处理，直接使用缓存文件")
        protr_reflection = np.load(f"{reflection_cache}/protr.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
        protr_reflection = protrGetter.mult_getter(df)
        np.save(f"{reflection_cache}/protr.npy", protr_reflection)

    result = df["UniProt_ID"].apply(lambda ID: protr_reflection[ID])
    result = pd.DataFrame(result.tolist(), columns=[f"protr{_}" for _ in range(1, result.iloc[0].shape[0] + 1)])
    return result


def param4MutTm(df: pd.DataFrame, reflection_cache: str) -> pd.DataFrame:
    print("-获取[param]特征", end=".....")
    if os.path.exists(f"{reflection_cache}/param.npy") or sys.platform == "linux":
        print("该数据集已经经过处理，直接使用缓存文件")
        param_reflection = np.load(f"{reflection_cache}/param.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
        param_getter = paramGetter.ParamGetter()
        param_reflection = param_getter.params_getter(df)
        np.save(f"{reflection_cache}/param.npy", param_reflection)

    result = df["UniProt_ID"].apply(lambda ID: param_reflection[ID])
    result = pd.DataFrame(result.tolist(), columns=[f"param{_}" for _ in range(1, result.iloc[0].shape[0] + 1)])
    # for the Ext.coffience
    mean_indexs = []
    na_indexs = []
    for index, coef in result["param5"].items():
        if not math.isclose(coef, 1e9):
            mean_indexs.append(index)
        else:
            na_indexs.append(index)
    mean_value = result["param5"].iloc[mean_indexs].mean()
    result["param5"] = result["param5"].astype(float)
    result.loc[na_indexs, "param5"] = mean_value
    return result


def hydrop4MutTm(df: pd.DataFrame, reflection_cache: str) -> pd.DataFrame:
    print("-获取[hydrop]特征", end=".....")
    if os.path.exists(f"{reflection_cache}/hydrop.npy"):
        print("该数据集已经经过处理，直接使用缓存文件")
        hydrop_reflection = np.load(f"{reflection_cache}/hydrop.npy", allow_pickle=True).item()
    else:
        print("该数据集未经过处理，重新处理并生成缓存文件")
        hydrop_getter = hydropGetter.ScaleGetter()
        hydrop_reflection = hydrop_getter.scale_getter(df)
        np.save(f"{reflection_cache}/hydrop.npy", hydrop_reflection)

    result = df["UniProt_ID"].apply(lambda ID: hydrop_reflection[ID])
    result = pd.DataFrame(result.tolist(), columns=[f"hydrop{_}" for _ in range(1, result.iloc[0].shape[0] + 1)])
    return result


def embedding4MutTm(df: pd.DataFrame,
                    embedding_model_path,
                    embedding_model_name,
                    embedding_method) -> pd.DataFrame:
    """
    :param embedding_method:
    :param df: 包含UniProt_ID, MutantSeq, WildSeq的DataFrame
    :param embedding_model_path: 嵌入模型路径
    :param embedding_model_name: 嵌入模型名称
    :return:
    """
    print("-获取[embedding]特征", end=".....")
    print("该特征没有缓存文件，直接生成...")
    device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")

    encoder = embeddingGetter.DeepEncoder(embedding_model_path, embedding_model_name, device)
    mutant_ref = encoder.encode_from_df_seperated(df[["UniProt_ID", "MutantSeq"]])
    wild_ref = encoder.encode_from_df_seperated(df[["UniProt_ID", "WildSeq"]])

    if embedding_method == "sub":
        embedding_reflection = {ID: mutant_ref[ID] - wild_ref[ID] for ID in mutant_ref.keys()}
    elif embedding_method == "cat":
        embedding_reflection = {ID: np.append(mutant_ref[ID], wild_ref[ID]) for ID in mutant_ref.keys()}
    elif embedding_method == "add":
        embedding_reflection = {ID: mutant_ref[ID] + wild_ref[ID] for ID in mutant_ref.keys()}

    encoder.release()
    print("-释放模型后显存用量:", torch.cuda.memory_allocated(device) / (1024 ** 3))

    result = df["UniProt_ID"].apply(lambda ID: embedding_reflection[ID])
    result = pd.DataFrame(result.tolist(), columns=[f"embedding{_}" for _ in range(1, result.iloc[0].shape[0] + 1)])
    return result


def context_embedding4MutTm(df: pd.DataFrame,
                            embedding_model_path,
                            embedding_model_name,
                            embedding_method) -> pd.DataFrame:
    """
    :param embedding_method:
    :param df: 包含UniProt_ID, MutantSeq, WildSeq的DataFrame
    :param embedding_model_path: 嵌入模型路径
    :param embedding_model_name: 嵌入模型名称
    :return:
    """
    print("-获取[context_embedding]特征,该特征没有缓存文件，直接生成...")
    device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")

    encoder = embeddingGetter.DeepEncoder(embedding_model_path, embedding_model_name, device)
    mutant_ref = encoder.encode_from_df_seperated(df[["UniProt_ID", "MutantCutSeq"]])
    wild_ref = encoder.encode_from_df_seperated(df[["UniProt_ID", "WildCutSeq"]])

    if embedding_method == "sub":
        embedding_reflection = {ID: mutant_ref[ID] - wild_ref[ID] for ID in mutant_ref.keys()}
    elif embedding_method == "cat":
        embedding_reflection = {ID: np.append(mutant_ref[ID], wild_ref[ID]) for ID in mutant_ref.keys()}
    elif embedding_method == "add":
        embedding_reflection = {ID: mutant_ref[ID] + wild_ref[ID] for ID in mutant_ref.keys()}
    else:
        raise ValueError("embedding_method must be 'sub', 'cat' or 'add'")

    encoder.release()
    print("-释放模型后显存用量:", torch.cuda.memory_allocated(device) / (1024 ** 3))

    result = df["UniProt_ID"].apply(lambda ID: embedding_reflection[ID])
    result = pd.DataFrame(result.tolist(), columns=[f"embedding{_}" for _ in range(1, result.iloc[0].shape[0] + 1)])
    return result


def GO4MutTm(train_df: pd.DataFrame,
             test_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    print("-获取[GO]特征", end=".....")
    # 提取Site与Ancestor特征
    train_df, test_df = GOGetter.ancestorGetter(train_df), GOGetter.ancestorGetter(test_df)
    train_df, test_df = GOGetter.siteGetter(train_df), GOGetter.siteGetter(test_df)
    # 计算PA值与LR值
    train_PA, test_PA = GOGetter.calculate_PA(train_df[["Positive", "site"]], test_df[["Positive", "site"]])
    train_LR, test_LR = GOGetter.calculate_LR(train_df[["Positive", "ancestor"]], test_df[["Positive", "ancestor"]])
    train_df = train_df.drop(columns=["site", "ancestor"])
    test_df = test_df.drop(columns=["site", "ancestor"])
    # return
    train_df.insert(loc=train_df.shape[1], column="PA", value=train_PA)
    test_df.insert(loc=test_df.shape[1], column="PA", value=test_PA)
    train_df.insert(loc=train_df.shape[1], column="LR", value=train_LR)
    test_df.insert(loc=test_df.shape[1], column="LR", value=test_LR)
    return train_df, test_df


class Dataset4MutTm:
    """
    用于创建MutTm-pred数据集的类，主要进行了以下工作：\n
    """

    def __init__(self,
                 package_path: str,
                 test_dataset_path: str,
                 train_dataset_path: str,
                 training_version: str,
                 testing_version: str = None,
                 selected_columns: list = None,
                 features=None,
                 context_length: int = 50,
                 embedding_model_path: str = None,
                 embedding_model_name: str = None,
                 embedding_method: str = "sub"
                 ):
        """
        参数类
        :param package_path: 包路径
        :param test_dataset_path: 训练数据集的绝对路径
        :param train_dataset_path: 测试数据集的绝对路径
        :param training_version: 训练数据集的版本[主要用于保存映射缓存文件]
        :param selected_columns: 选定的初始条目，默认值为["UniProt_ID", "Mutation", "Tm", "pH", "ΔTm"]
        :param mult_mode: 针对原始数据集中单一条目多测量值的处理方式，默认值为"Delete"
        :param features: 选定的特征集
        :param context_length: 上下文长度
        :param embedding_model_path: embedding模型路径
        :param embedding_model_name: embedding模型名称, 可用的模型组合如下\n
        ESM-2-650M -\n  DeepLearning/EmbeddingModels/ESM-2/esm2_t33_650M_UR50D \n
        ESM-2-3B -\n  DeepLearning/EmbeddingModels/ESM-2/esm2_t36_3B_UR50D \n
        ProtBert -\n  DeepLearning/EmbeddingModels/ProtBert \n
        CARF -\n
        """
        # set the package path
        self.package_path = package_path

        # train & test version
        self.training_version = training_version
        self.testing_version = testing_version

        # embedding model path & name
        self.embedding_model_path = embedding_model_path
        self.embedding_model_name = embedding_model_name
        self.embedding_method = embedding_method

        # select the columns
        self.selected_columns = selected_columns
        self.columns_without_label = selected_columns[:-1]
        self.basic_info = ["UniProt_ID", "Mutation", "ΔTm",
                           "ProteinSeq", "Fasta",
                           "Length",
                           "AminoAcidFrom", "MutationIndex", "AminoAcidTo", "Positive",
                           "WildSeq", "MutantSeq",
                           "WildCutSeq", "MutantCutSeq"]

        # reading the initializationDataset
        self.training_dataset = file2df(train_dataset_path)[self.selected_columns]
        self.testing_dataset = file2df(test_dataset_path)[self.selected_columns]

        # cache saving for debug
        self.cache_path = os.path.join(package_path, r"cache")
        self.training_reflection = os.path.join(package_path, rf"reflection_cache/{self.training_version}")
        self.testing_reflection = os.path.join(package_path, rf"reflection_cache/{self.testing_version}")

        # selected features
        self.features = features

        # basic, feature and label
        self.train_label_set = self.test_label_set = None
        self.train_basic_set = self.test_basic_set = None
        self.train_feature_set = self.test_feature_set = None

        # context length
        self.context_length = context_length

        # generate the testLoader
        self.pre_process()

    def show(self) -> None:
        # 打印所有对象属性
        pass

    def training_init(self):
        self.training_dataset, _ = wash4MutTm(df=self.training_dataset,
                                              focus_columns=self.columns_without_label,
                                              avg_filling=False)
        self.training_dataset = duplicate4MutTm(df=self.training_dataset,
                                                focus_columns=self.columns_without_label,
                                                mode="Average")
        self.training_dataset = delete4Dup(df=self.training_dataset,
                                           test_df=self.testing_dataset,
                                           focus_columns=self.columns_without_label)
        self.training_dataset = necessity4MutTm(df=self.training_dataset,
                                                refelction_cache=self.training_reflection,
                                                context_length=self.context_length)
        pass

    def testing_init(self):
        self.testing_dataset, _ = wash4MutTm(df=self.testing_dataset,
                                             focus_columns=self.columns_without_label,
                                             avg_filling=False)
        self.testing_dataset = duplicate4MutTm(df=self.testing_dataset,
                                               focus_columns=self.columns_without_label,
                                               mode="Average")
        self.testing_dataset = necessity4MutTm(df=self.testing_dataset,
                                               refelction_cache=self.testing_reflection,
                                               context_length=self.context_length)
        pass

    def feature_extract(self, ds: pd.DataFrame,
                        reflection_dirname: str,
                        dataset_version: str):
        features_list = list()
        # extract features
        if "neighbor" in self.features:
            features_list.append(neighbor4MutTm(df=ds, reflection_cache=reflection_dirname))
        if "aaindex" in self.features:
            features_list.append(aaindex4MutTm(df=ds, reflection_cache=reflection_dirname))
        if "group" in self.features:
            features_list.append(group4MutTm(df=ds, reflection_cache=reflection_dirname))
        if "sift4g" in self.features:
            features_list.append(
                sift4MutTm(df=ds, reflection_cache=reflection_dirname, dataset_version=dataset_version))
        if "param" in self.features:
            features_list.append(param4MutTm(df=ds, reflection_cache=reflection_dirname))
        if "rpm" in self.features:
            features_list.append(pssm4MutTm(df=ds, reflection_cache=reflection_dirname))
        if "swissrpm" in self.features:
            features_list.append(swisspssm4MutTm(df=ds, reflection_cache=reflection_dirname))
        if "protr" in self.features:
            features_list.append(protr4MutTm(df=ds, reflection_cache=reflection_dirname))
        if "hydrop" in self.features:
            features_list.append(hydrop4MutTm(df=ds, reflection_cache=reflection_dirname))
        if "embedding" in self.features:
            if self.embedding_model_path is None or self.embedding_model_name is None:
                raise ValueError("Please specify the embedding model path and name")
            features_list.append(embedding4MutTm(df=ds,
                                                 embedding_model_path=self.embedding_model_path,
                                                 embedding_model_name=self.embedding_model_name,
                                                 embedding_method=self.embedding_method))
        if "context_embedding" in self.features:
            if self.embedding_model_path is None or self.embedding_model_name is None:
                raise ValueError("Please specify the embedding model path and name")
            if self.context_length == 0:
                raise ValueError("Please specify the context length, and make sure if is btween 1 and 500")
            features_list.append(context_embedding4MutTm(df=ds,
                                                         embedding_model_path=self.embedding_model_path,
                                                         embedding_model_name=self.embedding_model_name,
                                                         embedding_method=self.embedding_method))

        # concat the features
        return pd.concat([ds] + features_list, axis=1)

    def pre_process(self):
        device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")
        print("当前使用设备： ", torch.cuda.get_device_name(device))

        print(f"===正在从训练集版本为{self.training_version}、"
              f"测试集版本为{self.testing_version}的原始数据集中"
              f"进行数据清洗和生物特征提取工作===")
        print("1.预处理训练集数据...")
        self.training_init()
        print("2.预处理测试集数据...")
        self.testing_init()
        print("3.为训练集数据提取生物特征...")
        self.training_dataset = self.feature_extract(self.training_dataset, self.training_reflection,
                                                     self.training_version)
        print("4.为测试集数据提取生物特征...")
        self.testing_dataset = self.feature_extract(self.testing_dataset,
                                                    self.testing_reflection,
                                                    self.testing_version)

        if "GO" in self.features:
            print("5.利用GO富集分析从训练集和测试集中提取LR与PA值")
            self.training_dataset, self.testing_dataset = GO4MutTm(train_df=self.training_dataset,
                                                                   test_df=self.testing_dataset)
        print("6.从全数据集中提取生物特征集、标签集和基本信息集...")
        self.train_basic_set = self.training_dataset[self.basic_info]
        self.train_feature_set = self.training_dataset.drop(columns=self.basic_info).astype(dtype="float64")
        self.train_label_set = self.training_dataset["ΔTm"].astype(dtype="float64")

        self.test_basic_set = self.testing_dataset[self.basic_info]
        self.test_feature_set = self.testing_dataset.drop(columns=self.basic_info).astype(dtype="float64")
        self.test_label_set = self.testing_dataset["ΔTm"].astype(dtype="float64")

        torch.cuda.empty_cache()  # 释放多余显存
        print(f"7.数据清洗和生物特征提取工作完成==>当前显存用量:{torch.cuda.memory_allocated(device) / (1024 ** 3)}")


if __name__ == "__main__":
    dataset = Dataset4MutTm(package_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Process4Dataset",
                            train_dataset_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\PonDB\pH-Tm\PonDB.csv",
                            test_dataset_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\ProThermDB\pH-Tm\excllent_ProThermDB_Testing.csv",
                            training_version="PonDB_withpHTm",
                            testing_version="ProThermDBTest_withpHTm",
                            selected_columns=["UniProt_ID", "Mutation", "pH", "Tm", "ΔTm"],
                            features=["neighbor", "aaindex", "group", "param", "swissrpm"],
                            context_length=25,
                            embedding_model_path="DeepLearning/EmbeddingModels/ESM-2/esm2_t33_650M_UR50D",
                            embedding_model_name="ESM-2-650M",
                            embedding_method="sub")
    global_train_x = dataset.train_feature_set
    global_train_y = dataset.train_label_set
    global_test_y = dataset.test_label_set
    global_test_x = dataset.test_feature_set
    x_train = np.array(global_train_x)
    x_test = np.array(global_test_x)
    y_train = np.array(global_train_y).reshape(len(global_train_y), 1).ravel()
    y_test = np.array(global_test_y).reshape(len(global_test_y), 1).ravel()
