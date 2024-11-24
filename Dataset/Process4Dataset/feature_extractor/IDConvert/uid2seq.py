# -*- coding : UTF-8 -*-
# @Author : JieJue
# @Project : PON-Pytorch副本[7.21]
# @Time : 2023/9/23 14:22
# @Name : Test
# @IDE : PyCharm
# @Introduction : as follows
"""
以下代码均来自 https://www.codenong.com/52569622/
采用两种方法处理出蛋白质对应的序列信息：
1.下载对应数据库的fasta文件，使用pyfaidx模块种的Fasta函数以提取文件种存储的序列
    {由于下载数据库过大，不便于项目运行，故放弃该方法}
2.利用蛋白质UniProtID，通过" http://www.uniprot.org/uniprot/+UniprotID+.fasta"访问对应序列信息
    {轻量级，适合项目}
"""
import time

import numpy as np
import urllib3
import requests
import pandas as pd
from Bio import SeqIO
from io import StringIO
from tqdm import tqdm
import os
urllib3.disable_warnings()
# 增加POST首部已避免Uniprot的反爬，并且阻止urllib3的报错以避免错误信息
header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# 缓存
abs_path = os.path.dirname(__file__)                                                               # 文件绝对路径
# abs_path = "D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Process4Dataset\feature_extractor\uid2seq.py"     # 当jupyter调用时可能无法使用__file__请自行修改
rel_path = os.path.relpath(abs_path).replace("\\", "/")                             # 相对/调用路径
uid_cache = os.path.join(rel_path, "uid/fasta.npy").replace("\\", "/")
wrong_id = []


def single_uid(uniprot_id: str) -> tuple:
    """
    通过UniProt官网获取蛋白质序列以及原始Fasta文件
    :param uniprot_id: 蛋白质的UniprotID
    :return: 返回序列以及原始Fasta文件
    """
    # region SeqGetter
    sequence_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    fasta_url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta?include=yes"

    sequence_response = requests.get(sequence_url)
    fasta_response = requests.get(fasta_url)

    if sequence_response.status_code == 200 and fasta_response.status_code == 200:
        # 提取序列部分（跳过标题行）
        sequence = ''.join(sequence_response.text.strip().split('\n')[1:])
        fasta = fasta_response.text.strip()
        return sequence, fasta
    else:
        wrong_id.append(uniprot_id)
        return None, None


def mult_uid(df: pd.Series
             ) -> dict:
    """
    利用UniprotID获取对应蛋白质序列
    :param df: 包含UniprotID的Series
    :param verbose: 是否打印调试信息
    :return: 返回{UniproID: 蛋白质序列}
    """
    history_cache = np.load(uid_cache, allow_pickle=True).item()  # 读取历史纪录
    id2seq_dict = {}
    wrong_id = []
    with tqdm(total=len(df)) as processingBar:
        for ID in df:
            if ID not in history_cache:
                # 由于uniprot的访问存在限制，会出现断连的情况，因此给予十次重连机会
                times = 0
                while times < 10:
                    try:
                        tSeq = single_uid(ID)
                        id2seq_dict[ID] = tSeq
                        break
                    except Exception:
                        times += 1
                        time.sleep(2)
                        print(f"Error happended in uidGetter -- {ID}: {times} times")
                        if times > 10:
                            raise Exception("十次连接失败，程序停止...")
            else:
                id2seq_dict[ID] = history_cache[ID]

            processingBar.update(1)

    print("===========Wrong uniprot_id:=========")
    for uniprot_id in wrong_id:
        print(f"A error happended in seqGetter -- No seq can reflect this uniprot_id:({uniprot_id}), please check this in UniProt")
    print("=======End of Wrong uniprot_id=======")

    # 保存历史纪录
    np.save(uid_cache, id2seq_dict)

    return id2seq_dict


# test here
if __name__ == '__main__':
    # result = single_uid("P11716")
    np.save(uid_cache, {None: (None, None)})
    pass

