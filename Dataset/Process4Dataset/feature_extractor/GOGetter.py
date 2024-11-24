import copy
import json
import math
import os
import time
from io import StringIO
import jsonpath
import numpy as np
import pandas as pd
import requests
import tqdm

abs_path = os.path.dirname(__file__)  # 文件绝对路径
# abs_path = "D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Process4Dataset\feature_extractor\GOGetter.py"     # 当jupyter调用时可能无法使用__file__请自行修改
rel_path = os.path.relpath(abs_path).replace("\\", "/")  # 相对/调用路径
site_cache = os.path.join(rel_path, "GO/site.npy").replace("\\", "/")
ancestor_cache = os.path.join(rel_path, "GO/ancestor.npy").replace("\\", "/")
ancestor_path = os.path.join(rel_path, "GO/Ancestor4GO.csv").replace("\\", "/")


def siteGetter(df: pd.DataFrame) -> pd.DataFrame:
    def calc(rawData, mutationIndex):
        lst = set()
        mutationIndex = int(mutationIndex)
        for pos in rawData.index:
            if rawData.loc[pos]["from"] <= mutationIndex <= rawData.loc[pos]["to"]:
                lst.add(rawData.loc[pos, "site"])

        return ",".join(list(lst))

    def gffGetter(ID):
        url = f"https://www.uniprot.org/uniprot/{ID}.gff"
        urlData = requests.get(url).content
        rowData = pd.read_csv(StringIO("\n".join(urlData.decode('utf-8').split('\n')[2:])),
                              sep="\t", header=None,
                              names=['uniprot_id', 'database', 'site', 'from', 'to', '1', '2', '3', '4', '5'])
        return rowData

    available_ref = np.load(site_cache, allow_pickle=True).item()  # 读取历史纪录
    df.insert(loc=df.shape[1], column="site", value="")
    with tqdm.tqdm(total=df.shape[0]) as procesingBar:  # 处理过的就直接读取，没有处理过的就重新处理
        for rowIndex, row in df.iterrows():
            label = row["UniProt_ID"] + str(row["MutationIndex"])
            # 判断是否已经处理过
            if label not in available_ref:
                # 由于uniprot的访问存在限制，会出现断连的情况，因此给予十次重连机会
                times = 0
                while True:
                    try:
                        result = calc(gffGetter(row["UniProt_ID"]), row["MutationIndex"])
                        available_ref[label] = result  # 写入字典
                        break
                    except Exception:
                        times += 1
                        if times > 10:
                            raise Exception("十次连接失败，程序停止...")
                        continue
            else:
                result = available_ref[label]

            df.at[rowIndex, "site"] = result
            procesingBar.update(1)

            if int(str(rowIndex)) != 0 and int(str(rowIndex)) % 10 == 0:
                np.save(site_cache, available_ref)

    np.save(site_cache, available_ref)  # 保存历史纪录
    return df


def ancestorGetter(df: pd.DataFrame) -> pd.DataFrame:
    """
    利用QuickGo网站对蛋白质进行GO分析
    :param df: 以UniprotID为列名的Series
    :return: {UniprotID: {"ancestor": [ancestor1, ancestor2, ..., ancestorn]}}, n >= 0
    """

    def get_GO(UID):
        response = requests.get("https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={}".format(UID),
                                headers={"Accept": "application/json"})

        # 如果对应GO.json不存在，则抛出异常
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(f"{UID} do not have a GO reflection in QuickGO. Please do it manuelly")

        # 读取爬取到的json数据
        responseBody = response.text
        responseBody = json.loads(responseBody)
        gos = jsonpath.jsonpath(responseBody, '$..goId')

        # 如果存在标记的话则提取所有可能存在的
        if gos:
            # 查找祖先
            ancestor = pd.read_csv(ancestor_path, sep=',', header=None, low_memory=False).set_index(0)

            lst = []
            for i in gos:
                if i in ancestor.index:
                    lst.extend(
                        ancestor.loc[i, :].tolist())  # TODO:Change into extend(l = l + ancestor.loc[i, :].tolist())
                lst.append(i)
            lst = list(set(lst))
            if np.nan in lst:
                lst.remove(np.nan)
            if 'all' in lst:
                lst.remove('all')
            gos = ','.join(lst)

            if isinstance(gos, bool):
                gos = ""
        # 返回获取到的GO标记
        return gos

    available_ref = np.load(ancestor_cache, allow_pickle=True).item()  # 读取历史纪录
    df.insert(loc=df.shape[1], column="ancestor", value="")
    with tqdm.tqdm(total=df.shape[0]) as procesingBar:  # 处理过的就直接读取，没有处理过的就重新处理
        for rowIndex, row in df.iterrows():
            if row["UniProt_ID"] not in available_ref:
                result = get_GO(row["UniProt_ID"])  # 重新处理
                available_ref[row["UniProt_ID"]] = result  # 写入字典
            else:
                result = available_ref[row["UniProt_ID"]]
            if isinstance(result, bool):
                result = ""

            df.at[rowIndex, "ancestor"] = result
            procesingBar.update(1)
    np.save(ancestor_cache, available_ref)  # 保存历史纪录
    return df


# 根据cv训练集计算所有的LR值
def calculate_LR(df1, df2):
    """
    df1:cv training set
    df2:cv test set
    """
    # log ((2+c)/(1+c)) + log ((2+c)/ (1+c)), {c==1}

    # 有害和中性注释的字典
    p = {}
    n = {}
    for index, row in df1.iterrows():
        if pd.isna(row["ancestor"]) or row["ancestor"] == "":
            continue
        if isinstance(row["ancestor"], bool):
            print("OK")
        for i in row["ancestor"].split(","):
            if i not in p.keys():
                p[i] = 1
                n[i] = 1
            if row["Positive"] == 1:
                p[i] += 1
            else:
                n[i] += 1

    for index, row in df2.iterrows():
        if pd.isna(row["ancestor"]):
            continue
        for i in row["ancestor"].split(","):
            if i not in p.keys():
                p[i] = 1
                n[i] = 1
            if row["Positive"] == 1:
                p[i] += 1
            else:
                n[i] += 1

    l = copy.deepcopy(p)
    for i in l.keys():
        l[i] = math.log(p[i] / n[i])

    # 求和计算每个蛋白的lr
    def LR_add(x):
        sump = 0
        if pd.isna(x):
            return sump
        for i in x.split(","):
            if i != "":
                sump += l[i]
        return sump

    return df1["ancestor"].apply(lambda x: LR_add(x)), df2["ancestor"].apply(lambda x: LR_add(x))


# 根据cv训练集计算所有的LR值
def calculate_PA(df1, df2):
    """
    df1:cv training set
    df2:cv test set
    """
    # log ((2+c)/(1+c)) + log ((2+c)/ (1+c)), {c==1}

    # 有害和中性注释的字典
    p = {}
    n = {}
    for index, row in df1.iterrows():
        if pd.isna(row["site"]) or row["site"] == "":
            continue
        for i in row["site"].split(","):
            if i != "":
                if i not in p.keys():
                    p[i] = 1
                    n[i] = 1
                if row["Positive"] == 1:
                    p[i] += 1
                else:
                    n[i] += 1
    for index, row in df2.iterrows():
        if pd.isna(row["site"]):
            continue
        for i in row["site"].split(","):
            if i != "":
                if i not in p.keys():
                    p[i] = 1
                    n[i] = 1
                if row["Positive"] == 1:
                    p[i] += 1
                else:
                    n[i] += 1

    s = copy.deepcopy(p)
    for i in s.keys():
        s[i] = math.log(p[i] / n[i])

    # 求和计算每个蛋白的pa
    def PA_add(x):
        sump = 0
        if pd.isna(x):
            return sump
        for i in x.split(","):
            if i != "":
                sump += s[i]
        return sump

    return df1["site"].apply(lambda x: PA_add(x)), df2["site"].apply(lambda x: PA_add(x))


def mult_getter(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # 提取指定列
    train_df = train_df[["UniProt_ID", "MutationIndex", "Positive"]]
    test_df = test_df[["UniProt_ID", "MutationIndex", "Positive"]]
    # 提取Site与Ancestor特征
    train_df, test_df = ancestorGetter(train_df), ancestorGetter(test_df)
    train_df, test_df = siteGetter(train_df), siteGetter(test_df)
    # 计算PA值与LR值
    calculate_PA(train_df, test_df)
    calculate_LR(train_df, test_df)
    return train_df, test_df


if __name__ == '__main__':
    res = np.load(site_cache, allow_pickle=True).item()
    pass
