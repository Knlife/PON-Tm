import os
import sys

import pandas as pd
from os.path import dirname, join

file_path = dirname(__file__)  # 只能在Pycharm使用，console使用需要进行配置


def aaindex_got_from_file() -> dict:
    """
    利用AAIndex数据库预处理文件，对氨基酸突变进行打分
    :param sr: 包含["amino_acid_from", "amino_acid_to"]列的DataFrame
    :return: {"".join([amino_acid_from, amino_acid_to]): [aaindex1, aaindex2, ..., aaindexn]}
    """
    a_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    a_set = set(a_list)
    aa_list = [i + j for i in a_list for j in a_list]
    aaindex_path = join(file_path, "aaindex/AAIndexMatrix.txt")

    def get_aaindex(amino_acid_from, amino_acid_to):
        if amino_acid_from not in a_set or amino_acid_to not in a_set:
            return {}
        res = aaindex.loc["{}{}".format(amino_acid_from, amino_acid_to), :]
        return res.to_list()

    # read the aaindex file
    aaindex = pd.read_csv(aaindex_path, sep="\t", header=None, names=["name"] + aa_list, index_col="name")
    aaindex = aaindex.T
    # drop duplicates according to the combination of amino_acid_from and amino_acid_to
    return {"".join([aap[0], aap[1]]): get_aaindex(aap[0], aap[1])
            for aap in aa_list}


def aaindex_name_getter():
    res = ["False"]
    aaindex_path = join(file_path, "aaindex/AAIndexMatrix.txt")
    with open(aaindex_path, "r") as file:
        for line in file.readlines():
            res.append(line.split()[0])
    return res


if __name__ == '__main__':
    result = aaindex_name_getter()
