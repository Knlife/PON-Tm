# -*- coding : UTF-8 -*-
# @Author : JieJue
# @Project : MutTm-pred
# @Time : 2023/12/1 10:17
# @Name : aaindexGetter
# @IDE : PyCharm
# @Introduction : as follows
"""
本程序提供以下两个功能，
    1.用于将AAIndex数据库提供的.txt数据转换为易于Python处理的.csv文件(截取自 https://github.com/tadorfer/AAIndex)
    2.根据用户输入Mutation/amino acid计算出子数据库提供的特征值
"""
import time

import pandas as pd
import numpy as np


# region txt2csv module
def conversion_aa1(data):
    "Converts raw AAIndex1 into useable Pandas DataFrame"

    # define column names and initialize dataframe
    col1 = ['Description']
    aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'label', 'V']
    columns = col1 + aa
    df = pd.DataFrame(data=[], columns=columns)

    # conversion by parsing text file line by line
    with open(data) as f:
        for i, line in enumerate(f):
            if line[0] == 'H':
                description = line.split()[1]
            if line[0] == 'I':
                tmp = i
            if 'tmp' in locals():
                if i == tmp + 1:
                    tmp1 = [description] + line.split()
                if i == tmp + 2:
                    tmp2 = line.split()
                    tmp_all = tmp1 + tmp2
                    tmp_all = pd.DataFrame([tmp_all], columns=columns)
                    df = df.append([tmp_all]).reset_index(drop=True)

    return df


def conversion_aa2(data):
    "Converts raw AAIndex2 into useable Pandas DataFrame"

    # define column names
    columns = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'label', 'V']

    MAX_ROW = 22
    MAX_COL = 21
    INDICES = 94
    arr = np.zeros((MAX_ROW, MAX_COL, INDICES))
    cnt = -1
    all_desc = []

    with open(data) as f:
        for i, line in enumerate(f):
            if line[0] == 'H':
                description = line.split()[1]
                all_desc.append(description)
                cnt += 1
            if line[0] == 'M':
                tmp = i
            if 'tmp' in locals():
                for aa in range(MAX_ROW):
                    if i == tmp + (aa + 1):
                        tmp_arr = line.split()
                        # replacing dashes with NaN
                        tmp_arr = [e.replace("-", "NaN") if len(e) == 1 else e for e in tmp_arr]
                        try:
                            float(tmp_arr[0])
                            arr[aa, :len(tmp_arr), cnt] = tmp_arr
                        except ValueError:
                            pass

    rows = [str(x) for x in range(22)]
    cols = [str(x) for x in range(21)]

    ext_desc = [[all_desc[i]] * 22 for i in range(INDICES)]
    flat_desc = [item for sublist in ext_desc for item in sublist]
    multind = pd.MultiIndex.from_arrays([flat_desc, rows * INDICES], names=['Description', 'Amino Acids'])

    # reshape 3D to 2D
    arr2D = arr.transpose(2, 0, 1).reshape(-1, arr.shape[1])

    df = pd.DataFrame({cols[i]: arr2D[:, i] for i in range(21)}, multind)

    return df


def conversion_aa3(data):
    "Converts raw AAIndex3 into useable Pandas DataFrame"

    # define column names
    columns = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'label', 'V']

    MAX_ROW = 20
    MAX_COL = 20
    INDICES = 47
    arr = np.zeros((MAX_ROW, MAX_COL, INDICES))
    cnt = -1
    all_desc = []

    # conversion by parsing text file line by line
    with open(data) as f:
        for i, line in enumerate(f):
            if line[0] == 'H':
                description = line.split()[1]
                all_desc.append(description)
                cnt += 1
            if line[0] == 'M':
                tmp = i
            if 'tmp' in locals():
                for aa in range(MAX_ROW):
                    if i == tmp + (aa + 1):
                        tmp_arr = line.split()
                        # replacing dashes with NaN
                        tmp_arr = [e.replace("-", "NaN") if len(e) == 1 else e for e in tmp_arr]
                        tmp_arr = [e.replace("NA", "NaN") for e in tmp_arr]
                        arr[aa, :len(tmp_arr), cnt] = tmp_arr

    ext_desc = [[all_desc[i]] * 20 for i in range(INDICES)]
    flat_desc = [item for sublist in ext_desc for item in sublist]
    multind = pd.MultiIndex.from_arrays([flat_desc, columns * INDICES], names=['Description', 'Amino Acids'])

    # reshape 3D to 2D
    arr2D = arr.transpose(2, 0, 1).reshape(-1, arr.shape[1])

    df = pd.DataFrame({columns[i]: arr2D[:, i] for i in range(20)}, multind)

    return df


# endregion


# region extraction module
aaindex1 = pd.read_csv("../../aaindex/aaindex1.csv")
aaindex2 = pd.read_csv("../../aaindex/aaindex2.csv")
aaindex3 = pd.read_csv("../../aaindex/aaindex3.csv")
aminoacids = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
              "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "label": 18, "V": 19}


# Extract the physical and chemical MachineLearning from AAindex1
def aminoacidGetter(td: pd.Series,
                    selected_codes: list = None,
                    verbose: bool = True) -> dict:
    if selected_codes is None:
        selected_codes = []

    def aminoacidGotFromIO(acid_name: str) -> list:
        selection = set(selected_codes)
        occur_list = []
        if len(selection) == 0:
            all_flag = True
        else:
            all_flag = False

        for rowindex, row in aaindex1.iterrows():
            if all_flag or row["Description"] in selection:
                occur_list.append(row[acid_name])
        return occur_list

    occurs = list(set(list(td)))  # get the duplicated set
    if verbose:
        print(f"The number of non-duplicated and available aminoacid is {len(occurs)}")
    return {aminoacid: aminoacidGotFromIO(aminoacid) for aminoacid in occurs}


# Extract the mutation MachineLearning from AAindex2
def mutationInfoGetter(td: pd.DataFrame,
                       selected_codes: list = None,
                       verbose: bool = True) -> dict:
    if selected_codes is None:
        selected_codes = []

    def mutationInfoGotFromIO(pre: str,
                              cur: str
                              ) -> list:
        selection = set(selected_codes)
        occur_list = []
        if len(selection) == 0:
            all_flag = True
        else:
            all_flag = False

        for rowindex, row in aaindex2.iterrows():
            if all_flag or row["Description"] in selection:
                if aminoacids[pre] >= aminoacids[cur]:
                    if row["Amino Acids"] == aminoacids[pre]:
                        if verbose:
                            print(row["Description"], row["Amino Acids"], "==", pre, row.iloc[aminoacids[cur]])

                        occur_list.append(row[str(aminoacids[cur])])
                else:
                    if row["Amino Acids"] == aminoacids[cur]:
                        if verbose:
                            print(row["Description"], row["Amino Acids"], "==", cur, row.iloc[aminoacids[pre]])

                        occur_list.append(row[str(aminoacids[pre])])

        return occur_list

    occurs = list(set(list(td)))  # get the duplicated set
    if verbose:
        print(f"The number of non-duplicated and available mutation is {len(occurs)}")
    return {mutation: mutationInfoGotFromIO(mutation[0], mutation[-1]) for mutation in occurs}


# Extract protein pairwise contact potentials from AAindex3
def potentialGetter(td: pd.DataFrame,
                    selected_codes: list = None,
                    verbose: bool = True) -> dict:
    if selected_codes is None:
        selected_codes = []

    def potentialGotFromIO(pre: str,
                           cur: str
                           ) -> list:
        selection = set(selected_codes)
        occur_list = []
        if len(selection) == 0:
            all_flag = True
        else:
            all_flag = False

        for rowindex, row in aaindex3.iterrows():
            if all_flag or row["Description"] in selection:
                if row["Description"] == "ROBB790102":
                    continue
                if aminoacids[pre] >= aminoacids[cur]:
                    if row["Amino Acids"] == pre:
                        if verbose:
                            print(row["Description"], row["Amino Acids"], "==", pre, row[cur])
                        occur_list.append(row[cur])
                else:
                    if row["Amino Acids"] == cur:
                        if verbose:
                            print(row["Description"], row["Amino Acids"], "==", cur, row[pre])
                        occur_list.append(row[pre])

        return occur_list

    occurs = list(set(list(td)))  # get the duplicated set
    if verbose:
        print(f"The number of non-duplicated and available aminoacid is {len(occurs)}")
        print(2333)
    return {pair: potentialGotFromIO(pair[0], pair[-1]) for pair in occurs}
# endregion
