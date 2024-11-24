import pandas as pd


def single_getter(aminoacid_from, aminoacid_to):
    reflections = {'A': 5, 'C': 0, 'D': 1, 'E': 1, 'F': 0, 'G': 3, 'H': 2, 'I': 0, 'K': 2, 'L': 0,
                   'M': 0, 'N': 4, 'P': 3, 'Q': 4, 'R': 2, 'S': 4, 'T': 5, 'V': 0, 'W': 0, 'Y': 0}
    result = [0 for _ in range(36)]
    result[reflections[aminoacid_from.upper()] * 6 + reflections[aminoacid_to.upper()]] = 1
    return result


def mult_getter(df: pd.DataFrame) -> dict:
    """
    Group信息表征了从某一Group突变到Group的影响，以下为映射表\n
    Group1 : ['V', 'I', 'L', 'F', 'M', 'W', 'label', 'C'],  --> hydrophobic\n
    Group2 : ['D', 'E'],                                --> negatively charged\n
    Group3 : ['R', 'K', 'H'],                           --> positively charged\n
    Group4 : ['G', 'P'],                                --> conformational\n
    Group5 : ['N', 'Q', 'S'],                           --> polar\n
    Group6 : ['A', 'T']                                 --> others\n
    :param df: 传入的突变组合序列
    :return: 返回一个{"突变组合0": Group信息0(36 sr 1), ...., "突变组合n": Group信息n(36 sr 1)}
    """
    return {row["AminoAcidFrom"] + row["AminoAcidTo"]: single_getter(row["AminoAcidFrom"], row["AminoAcidTo"])
            for rowIndex, row
            in df[["AminoAcidFrom", "AminoAcidTo"]].iterrows()}
