import numpy as np
import pandas as pd
import re


def file2df(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", header=0)
    else:
        raise Exception(f"Unsupported file type: {path}, Expected .csv or .tsv")


def wash4MutTm(df: pd.DataFrame,
               avg_filling: bool,
               focus_columns: list) -> (pd.DataFrame, pd.DataFrame):
    """
    用于从DataFrame中读取columns包含的数据并进行清洗工作：\n
    1.为统一化数据格式，为DataFrame添加"pH"和"Tm"条目[filling]\n
    2.Mutation条目中可能存在多余的空格[MutWasher]\n
    3.deltaTm中可能存在以计算式表示的数据[deltaTmWasher]\n
    4.Tm中可能存在类似于“>95”“63(0.5)”的数据[TmWasher]\n
    5.pH中可能存在以计算式表示的数据[pHWasher]\n
    6.剔除全条目中可能存在的na/null/不合法数值字母[validation]
    :param avg_filling:
    :param focus_columns:
    :param df: 输入的pandas.Dataframe
    :return: 返回处理后的的pandas.DataFrame
    """

    # 突变信息分离
    def mutationCut(mutation: str) -> tuple:
        re_search = re.search(r"([A-Za-z]{1})(\d+)([A-Za-z]{1})", mutation)
        if re_search is None:
            return "a", "None", "a"
        aminoacid_from = re_search.group(1)
        aminoacid_index = re_search.group(2)
        aminoacid_to = re_search.group(3)
        return aminoacid_from, aminoacid_index, aminoacid_to

    # 针对Mutation异常数据的清洗
    def mutWasher(s: str) -> str:
        if "(" in s:  # 删除携带的附带信息
            s = s[:s.index("(")]
        s = s.strip(" ")  # 删除多余空格
        if s[0].islower() or s[0].isnumeric():
            s = s[s.index(":") + 1:]
        return "".join(mutationCut(s))

    # 针对ΔTm数据的清洗
    def deltaTmWasher(s) -> float:
        if isinstance(s, float) or isinstance(s, int):
            return s
        if isinstance(s, str) and "(" in s:
            return eval(s[:s.index("(")])
        return eval(s)

    # 针对Tm异常数据的清洗
    def TmWasher(s) -> float:
        if pd.isna(s):
            return s
        if isinstance(s, float) or isinstance(s, int):
            return s
        if isinstance(s, str) and "(" in s:
            return eval(s[:s.index("(")])
        return eval(s)

    # 针对pH异常数据的清洗
    def pHWasher(s) -> float:
        if pd.isna(s):
            return np.nan
        if isinstance(s, float):
            return s
        if isinstance(s, str) and "[" in s:
            return float(s[s.index("[") + 1: s.index("]")])
        else:  # 存在以计算式表示的pH
            return eval(s)

    # main
    aminoacid_set = {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                     'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'}
    deleteIndex = []

    # judge
    for rowIndex, row in df.iterrows():
        # 针对UniProt_ID异常数据的清洗
        if row["UniProt_ID"] == "-" or pd.isnull(row["UniProt_ID"]) or pd.isna(row["UniProt_ID"]):
            deleteIndex.append(rowIndex)
            continue

        # 针对Mutation异常数据的清洗
        if row["Mutation"] == "-" or pd.isnull(row["Mutation"]) or pd.isna(row["Mutation"]):
            deleteIndex.append(rowIndex)
            continue

        # 针对Mutation异常数据的清洗--非法突变
        if row["Mutation"] is not None:
            flag = False
            for item in mutationCut(row["Mutation"]):
                # 删除非常规残基
                if item.isalpha() and item not in aminoacid_set:
                    deleteIndex.append(rowIndex)
                    flag = True
                    break
            if flag:
                continue

        # 针对pH异常数据的清洗--后续可以用平均值处理
        if "pH" in focus_columns:
            if isinstance(row["pH"], str):
                if "-" in row["pH"]:
                    df.loc[rowIndex, "pH"] = np.nan
                elif ">" in row["pH"]:
                    deleteIndex.append(rowIndex)
                    continue
            elif pd.isnull(row["pH"]) or pd.isna(row["pH"]):
                df.loc[rowIndex, "pH"] = np.nan

        # 针对Tm异常数据的清洗--后续可以用平均值处理
        if "Tm" in focus_columns:
            if isinstance(row["Tm"], str):
                if "-" in row["Tm"]:
                    df.loc[rowIndex, "Tm"] = np.nan
                elif ">" in row["Tm"]:
                    deleteIndex.append(rowIndex)
                    continue
            elif pd.isnull(row["Tm"]) or pd.isna(row["Tm"]):
                df.loc[rowIndex, "Tm"] = np.nan

        # 针对PDB_Chain_Mutation
        if "PDB_Chain_Mutation" in focus_columns:
            if row["PDB_Chain_Mutation"] == "-" or pd.isnull(row["PDB_Chain_Mutation"]) or pd.isna(
                    row["PDB_Chain_Mutation"]):
                deleteIndex.append(rowIndex)
                continue

        # 针对ΔTm异常数据的清洗--标签不可以使用平均值进行处理
        if row["ΔTm"] == "-" or pd.isnull(row["ΔTm"]) or pd.isna(row["ΔTm"]):
            deleteIndex.append(rowIndex)
            continue

    print(f"-删除数据缺失行及非法行共计{len(deleteIndex)}行")
    delete_df = df.loc[deleteIndex, :]
    df = df.drop(index=deleteIndex).reset_index(drop=True)

    # 数据形式正则化
    df["Mutation"] = df["Mutation"].apply(mutWasher)  # Mutation
    df.loc[:, "ΔTm"] = df["ΔTm"].apply(deltaTmWasher)  # deltaTm
    if "Tm" in focus_columns:
        df.loc[:, "Tm"] = df["Tm"].apply(TmWasher)  # Tm
        df["Tm"].astype(dtype=np.float64)
    if "pH" in focus_columns:
        df.loc[:, "pH"] = df["pH"].apply(pHWasher)  # pH
        df["pH"].astype(dtype=np.float64)

    # 是否使用平均值填补缺失值
    if avg_filling:
        print("-补全pH/Tm的缺失值", end=" ")
        if "Tm" in focus_columns:
            # df["Tm"].fillna(Tm_mean)
            # with pd.option_context("future.no_silent_downcasting", True):
            #     df["Tm"] = df["Tm"].apply(Tm_predict).infer_objects(copy=False)
            print("-利用PonStab2.0对蛋白质序列的Tm进行预测补全...暂时未进行，因此对所有Tm缺失条目进行删除")
            pass

        if "pH" in focus_columns:
            pH_mean = df["pH"].mean()
            print(f"-对于实验条件pH，利用平均值进行补充，而pH的平均值为{pH_mean:.5f}", end=" ")
            # df["pH"].fillna(pH_mean)
            with pd.option_context("future.no_silent_downcasting", True):
                df["pH"] = df["pH"].fillna(pH_mean).infer_objects(copy=False)

        df = df.dropna().reset_index(drop=True)
        print("-由于热稳定性预测工具未完善，Tm的补全措施取消，剔除所有非法Tm")
    else:
        print("-丢弃pH/Tm的缺失值")
        naIndex = df[df.isna().any(axis=1)].index
        df = df.drop(index=naIndex).reset_index(drop=True)

    return df, delete_df


def duplicate4MutTm(df: pd.DataFrame,
                    focus_columns: list,
                    mode: str = "Delete",
                    ) -> pd.DataFrame:
    """
    针对输入的pandas.Dataframe进行去重
    :param focus_columns:
    :param df: 输入pandas.DataFrame，当df_path存在时失效
    :param mode: 针对多测量值的处理模式，Delete表示全部删除，Keep表示全部保留，Average表示多测量值取平均值合并为一项
    :return: 返回处理后的的pandas.DataFrame
    """

    if mode == "Delete":
        df = df.drop_duplicates(subset=focus_columns, keep=False)
    elif mode == "Average":
        df = df.groupby(focus_columns).mean().reset_index()
    elif mode == "Keep":
        pass
    else:
        raise Exception("模式错误，请输入Delete/Average/Keep中的一种处理模式")

    # output
    return df


def check4mult(df: pd.DataFrame,
               focus_colmns: list) -> (int, list):
    # def pattern(s: pd.Series) -> str:
    #     path = []
    #     for subS in s:
    #         if not isinstance(subS, str):
    #             path.append(str(subS))
    #         else:
    #             path.append(subS)
    #     return "-".join(path)
    #
    # # if ΔTm in focus_columns, remove it.
    # if "ΔTm" in focus_colmns:
    #     focus_colmns.remove("ΔTm")
    #
    # indexs = []
    # cnt = 0
    # left, right = 0, 1  # 左、右指针
    # while right < df.shape[0]:
    #     # 如果遇到了多测量值
    #     if pattern(df.loc[left, focus_colmns]) == pattern(df.loc[right, focus_colmns]):
    #         while right < df.shape[0] and pattern(df.loc[left, focus_colmns]) == pattern(df.loc[right, focus_colmns]):
    #             indexs.append(right)
    #             right += 1
    #         cnt += 1
    #         # 左指针指向下一个可能的重复位点
    #         left = right
    #         # 右指针指向下一个可能和左指针相同的位点
    #         right += 1
    #     # 没遇到就两个指针往前走
    #     else:
    #         left += 1
    #         right += 1

    duplicated_df = df[df.duplicated(subset=focus_colmns, keep=False)]
    return duplicated_df.shape[0], duplicated_df.index


def delete4Dup(df: pd.DataFrame,
               test_df: pd.DataFrame,
               focus_columns: list) -> pd.DataFrame:
    """
    :param focus_columns:
    :param test_df:
    :param df:
    :return:
    """

    def CreateDict(_df: pd.DataFrame) -> set:
        occur = set()
        for rowIndex, row in _df.iterrows():
            occur.add("@".join(map(str, row)))
        return occur

    def check(_df: pd.DataFrame, occur: set) -> pd.DataFrame:
        deleteIndex = []
        for rowIndex, row in _df[focus_columns].iterrows():
            if "@".join(map(str, row)) in occur:
                deleteIndex.append(rowIndex)

        print("-前一数据集采用了后一数据集中的{}条数据，现已删除".format(len(deleteIndex)))
        return _df.drop(index=deleteIndex).reset_index(drop=True)

    # main
    mtdData = CreateDict(test_df[focus_columns])
    return check(df, mtdData)


def select4Dup(df: pd.DataFrame,
               test_df: pd.DataFrame,
               focus_columns: list) -> pd.DataFrame:
    """
    :param focus_columns:
    :param test_df:
    :param df:
    :return:
    """

    def CreateDict(_df: pd.DataFrame) -> set:
        occur = set()
        for rowIndex, row in _df.iterrows():
            occur.add("@".join(map(str, row)))
        return occur

    def check(_df: pd.DataFrame, occur: set) -> pd.DataFrame:
        deleteIndex = []
        for rowIndex, row in _df[focus_columns].iterrows():
            if "@".join(map(str, row)) in occur:
                deleteIndex.append(rowIndex)

        print("-前一数据集采用了后一数据集中的{}条数据，现返回该重叠数据集".format(len(deleteIndex)))
        return _df.iloc[deleteIndex, :].reset_index(drop=True).copy()

    # main
    mtdData = CreateDict(test_df[focus_columns])
    return check(df, mtdData)


def cutter(sequence: str,
           residue_index: int,
           context_length: int) -> str:
    # region Initialize the variable
    index = residue_index  # 突变在序列字符串中的位置
    left = context_length // 2  # 突变上文
    right = (context_length - 1) // 2  # 突变下文
    # endregion

    # region Check the boundary => 0 <=  index + left < index + right < length
    if index + right >= len(
            sequence):  # if right is out of range => [index - context_length, index] + index + [index + 1, length - 1]
        cut_seq = sequence[len(sequence) - context_length:]
    elif index - left < 0:  # if left is out of range => [0, index] + index + [index + 1, context_length + 1]
        cut_seq = sequence[: context_length]
    else:  # if left and right are in range => [index - left, index] + index + [index + 1, index + right + 1]
        cut_seq = sequence[index - left: index + right + 1]
    # endregion

    return cut_seq


# get the context
def sequence_cutter(df: pd.DataFrame,
                    context_length: int) -> (pd.Series, pd.Series, pd.Series, pd.Series):
    # region Vertification
    if "AminoAcidTo" not in df.columns:
        raise Exception("缺少AminoAcidTo列")
    if "ProteinSeq" not in df.columns:
        raise Exception("缺少ProteinSeq列")
    if "MutationIndex" not in df.columns:
        raise Exception("缺少MutationIndex列")
    # endregion

    wild_sequences = []
    mutant_sequences = []
    wild_cut_sequences = []
    mutant_cut_sequences = []
    for rowIndex, row in df[["AminoAcidTo", "ProteinSeq", "MutationIndex"]].iterrows():
        wild_sequence = row["ProteinSeq"]
        mutant_index = int(row["MutationIndex"]) - 1  # protein index begin from 1, while str index begin from 0
        mutant_sequence = wild_sequence[: mutant_index] + row["AminoAcidTo"] + wild_sequence[mutant_index + 1:]
        wild_sequences.append(wild_sequence)
        mutant_sequences.append(mutant_sequence)
        if context_length:
            wild_cut_sequences.append(cutter(wild_sequence, mutant_index, context_length))
            mutant_cut_sequences.append(cutter(mutant_sequence, mutant_index, context_length))
        else:
            wild_cut_sequences.append("")
            mutant_cut_sequences.append("")
    return pd.Series(wild_sequences), pd.Series(mutant_sequences), pd.Series(wild_cut_sequences), pd.Series(
        mutant_cut_sequences)


def unipdb_getter(file_path: str):
    def pattern(pcm: str) -> (str, str, str):
        split_pattern = re.split("[:|_]", pcm)
        if len(split_pattern) != 3:
            return "", "", ""
        else:
            return split_pattern[0], split_pattern[1], split_pattern[2]

    # data read&wash
    df = file2df(file_path)[["UniProt_ID", "Mutation", "ΔTm", "PDB_Chain_Mutation"]]
    df.sort_values(by=["UniProt_ID", "Mutation", "PDB_Chain_Mutation"], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    df, _ = wash4MutTm(df,
                       avg_filling=False,
                       focus_columns=["UniProt_ID", "Mutation", "PDB_Chain_Mutation", "ΔTm"])

    # calc
    unipdb = {}
    state = {}
    for rowIndex, row in df.iterrows():
        uid = row["UniProt_ID"] + ":" + row["Mutation"]
        pdb_chain_mutation = row["PDB_Chain_Mutation"]
        if uid not in unipdb:
            unipdb[uid] = pdb_chain_mutation
            state[uid] = True
        else:
            if state[uid]:
                if pdb_chain_mutation != unipdb[uid]:
                    unipdb.pop(uid)
                    state[uid] = False

    print("-该数据集中仅有",
          len(unipdb),
          "/",
          len(df["UniProt_ID"]),
          "的数据可以再在此数据集背景下唯一映射PDB_ID")
    return unipdb


if __name__ == "__main__":
    gr = unipdb_getter(r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\InitializationDataset\ProThermDB.tsv")
