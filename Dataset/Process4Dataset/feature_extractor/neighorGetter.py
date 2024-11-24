from collections import defaultdict
from typing import Union
import pandas as pd

a_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'label']


def neighbor_getter(df: pd.DataFrame) -> dict:
    """
    根据输入Dataframe包含的蛋白质UniprotID、对应序列以及突变位点信息计算出突变位点周围25格内的邻域信息
    :param df: 存在列["UniProt_ID", "MutationIndex", "ProteinSeq"]的DataFrame
    :return: 返回{"UniProt_ID" + "-" + "MutationIndex": neighborhood_features}的dict
    """

    def get_neighborhood_features(seq, aminoacid_index):
        def child_seq(pseq: str,
                      muti: Union[int, float]):
            """
            :param pseq: 蛋白质序列
            :param muti: 突变位点
            :return: 返回突变位点相邻的序列
            """
            index = int(muti) - 1
            if index < 11:
                beg = 0
            else:
                beg = index - 11
            if index > len(pseq) - 13:
                end = len(pseq)
            else:
                end = index + 12
            return pseq[beg: end]

        def get_count_a(p):
            a_dict = defaultdict(int)
            for i in p:
                a_dict[i] += 1  # 递增
            return {'AA20D.' + i: a_dict[i] for i in a_list}

        piece = child_seq(seq, aminoacid_index)  # 获取突变邻域序列
        nei_feature = get_count_a(piece)  # 获取{氨基酸种类: 个数}统计字典
        # 1. NonPolarAA: Number of non-polar neighborhood residues
        nei_feature['NonPolarAA'] = (nei_feature['AA20D.' + a_list[0]]
                                     + nei_feature['AA20D.' + a_list[4]]
                                     + nei_feature['AA20D.' + a_list[5]]
                                     + nei_feature['AA20D.' + a_list[7]]
                                     + nei_feature['AA20D.' + a_list[9]]
                                     + nei_feature['AA20D.' + a_list[10]]
                                     + nei_feature['AA20D.' + a_list[12]]
                                     + nei_feature['AA20D.' + a_list[17]]
                                     + nei_feature['AA20D.' + a_list[18]]
                                     + nei_feature['AA20D.' + a_list[19]])
        # 2.PolarAA:Number of polar neighborhood residues
        nei_feature['PolarAA'] = (nei_feature['AA20D.' + a_list[1]]
                                  + nei_feature['AA20D.' + a_list[11]]
                                  + nei_feature['AA20D.' + a_list[13]]
                                  + nei_feature['AA20D.' + a_list[15]]
                                  + nei_feature['AA20D.' + a_list[16]])
        # 3.ChargedAA:Number of charged neighborhood residues
        nei_feature['ChargedAA'] = (nei_feature['AA20D.' + a_list[2]]
                                    + nei_feature['AA20D.' + a_list[3]]
                                    + nei_feature['AA20D.' + a_list[6]]
                                    + nei_feature['AA20D.' + a_list[8]]
                                    + nei_feature['AA20D.' + a_list[14]])
        # 4.PosAA:Number of Positive charged neighborhood residues
        nei_feature['PosAA'] = (nei_feature['AA20D.' + a_list[2]]
                                + nei_feature['AA20D.' + a_list[3]])
        # 5.NegAA:Number of Negative charged neighborhood residues
        nei_feature['NegAA'] = (nei_feature['AA20D.' + a_list[6]]
                                + nei_feature['AA20D.' + a_list[8]]
                                + nei_feature['AA20D.' + a_list[14]])
        return list(nei_feature.values())

    # drop duplicates according to UniProt_ID and MutationIndex for the same protein and same mutation index
    try:
        df = df.drop_duplicates(subset=["UniProt_ID", "MutationIndex"])[["UniProt_ID", "MutationIndex", "ProteinSeq"]]
    except Exception as e:
        print(e)
        raise Exception("缺少列：['UniProt_ID', 'MutationIndex', 'ProteinSeq']")

    # get neighborhood features
    return {str(row["UniProt_ID"]) + "-" + str(int(row["MutationIndex"]))
            : get_neighborhood_features(str(row["ProteinSeq"]), int(row["MutationIndex"]))
            for rowindex, row in df.iterrows()}
