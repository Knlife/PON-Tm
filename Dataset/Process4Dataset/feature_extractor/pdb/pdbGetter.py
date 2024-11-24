# -*- coding : UTF-8 -*-
# @Author : JieJue
# @Project : MutTm-pred
# @Time : 2023/12/11 23:26
# @Name : pdbGetter
# @IDE : PyCharm
# @Introduction : as follows
"""
利用AlphaFoldDB进行的从UniProtID->pdbID的程序
PS：由于AlphaFoldDB均为DeepMind团队使用AlphaFold模型预测的蛋白质结构，存在一定误差，请务必关注自己项目的逻辑问题！！！
"""
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio import PDB
import re
import requests
import urllib3
urllib3.disable_warnings()

header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


def pdbGetter(to_dir_path: str = None,
              id_list: list | pd.Series | np.ndarray = None,
              fasta_file_path: str = None):
    """
    根据给定形式的UniProt获取AlphaFoldDB下的.pdb文件，并将文件存储在to_dir_path目录下
    :param fasta_file_path:存储UniprotID获取的包含蛋白质序列信息的.fasta文件
    :param to_dir_path:存储.pdb文件的目录
    :param id_list:直接给定包含UniprotID的序列，list | pd.Series | np.ndarray三种格式可选，与fasta_file_path同时存在（一般不会出现）时优先被选用
    :return:
    """
    def fasta2list():
        ids = []
        with open(fasta_file_path, 'r') as fp:
            for line in fp:
                if line.startswith('>'):  # 作用：判断字符串是否以指定字符或子字符串开头
                    if line.startswith("sp") or line.startswith("tr"):
                        ids.append(re.match(r">\w{2}\|(\w+)\|.+", line).group(1))
                    else:
                        ids.append(line[1: -1])
        return ids

    def uniprot2pdb(IDs: list):

        not_exist_list = []
        for ID in IDs:
            print(ID, end=" -finding in-> ")
            url = 'https://alphafold.ebi.ac.uk/files/AF-' + ID + '-F1-model_v4' + '.pdb'
            print(url)

            response = requests.get(url, headers=header, verify=False)
            with open(f"{to_dir_path}/]" + ID + '.pdb', 'w') as files:

                response = response.text.splitlines()
                for lines in response:
                    files.write(lines)
                    files.write('\n')

            if response[0][1] == '?':

                not_exist_list.append(ID)

        print("以下ID于AlphaFoldDB中不存在对应的PDB文件，请自行到常用数据库检索...")
        [print(ID, end=" ") for ID in not_exist_list]
        return

    # Entrance
    if id_list:
        if isinstance(id_list, pd.Series):
            if id_list.shape[0] != 1 and id_list.shape[1] != 1:
                raise RuntimeError("Please input the right format Series, make sure the shape of data is single dimension, for example [1, 10] or [10, 1]")
            else:
                if id_list.shape[0] == 1:
                    return uniprot2pdb(list(id_list.iloc[, :]))
                else:
                    return uniprot2pdb(list(id_list.T.iloc[, :]))
        if isinstance(id_list, np.ndarray):
            if id_list.shape[0] != 1 and id_list.shape[1] != 1:
                raise RuntimeError(
                    "Please input the right format Ndarray, make sure the shape of data is single dimension, for example [1, 10] or [10, 1]")
            else:
                if id_list.shape[0] == 1:
                    return uniprot2pdb(list(id_list[0, :]))
                else:
                    return uniprot2pdb(list(id_list[:, 0]))
        if isinstance(id_list, list):
            return uniprot2pdb(id_list)
    elif fasta_file_path:
        return uniprot2pdb(fasta2list())
    else:
        raise RuntimeError("Please at least, input the .fasta file path or a list of UniprotID")


