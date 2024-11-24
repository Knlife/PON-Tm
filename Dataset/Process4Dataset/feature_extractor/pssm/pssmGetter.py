# -*- coding : UTF-8 -*-
# @Author : JieJue
# @Project : PON-Pytorch副本[7.21]
# @Time : 2023/9/23 16:39
# @Name : RpmGenerator
# @IDE : PyCharm
# @Introduction : as follows
"""
"""
import re
import subprocess
from io import StringIO

import numpy as np
import pandas as pd
import os
import re
import requests
import codecs  # 或者io，使用哪种包无所谓
import urllib3
from Bio import SeqIO

# R_HOME在Windows上设置有问题，必须在此处进行设置后才能调用rpy2
os.environ['R_HOME'] = r"D:\Program Files\R\R-4.3.2"
from rpy2 import robjects
robjects.r("""
library(PSSMCOOL)

ab_pssm <- function (pssm_path){
    return(AB_PSSM(pssm_name=pssm_path))
}
rpm_pssm <- function (pssm_path){
    return(RPM_PSSM(pssm_name=pssm_path))
}
""")

# url setting
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
urllib3.disable_warnings()


class PssmGetter:
    """
    Class for getting PSSM from Uniprot
    """
    def __init__(self,
                 cache_dir: str,
                 database_path: str,
                 verbose: bool = False):
        """
        利用PSSMCOOL从Uni_ProtID获取PSSM-varaion时，
        由于需要读写文件以及运行psiblast。
        因此需要传入存放临时文件的目录cache_dir，
        和运行psiblast的.fasta数据集文件路径database_path
        :param cache_dir:
        :param database_path:
        :param verbose:
        """
        self.cache_dir = cache_dir
        self.database_path = database_path
        self.verbose = verbose

    def get(self,
            ID: str,
            cData: str = None,
            variation: str = "None"):

        # 写入fasta文件，若不存在则从Uniprot获取
        if cData is None:
            if self.verbose:
                print("ProteinSeq info lost, getting from Uniprot.org!")
            with requests.post("https://www.uniprot.org/uniprot/" + ID + ".fasta",
                               headers=header,
                               verify=False) as response:
                cData = "".join(response.text)
        with open(fr"{self.cache_dir}/fastaFile.txt", "w") as basic:
            basic.write(cData)

        # 运行psiblast得到对应PSSM标准矩阵
        command = "psiblast {} {} {} {} {} {} {}".format("-comp_based_stats " + "1",
                                                         "-db " + self.database_path,
                                                         "-query " + self.cache_dir + "/fastaFile.txt",
                                                         "-evalue " + "0.001",
                                                         "-num_iterations " + "3",
                                                         "-num_threads " + "20",
                                                         "-out_ascii_pssm " + self.cache_dir + "/RPipe.pssm")
        result = subprocess.run(command, shell=True)
        if result.returncode:   # 若命令报错则抛出异常
            raise Exception(f"Psiblast failed in {ID}")

        return np.array(robjects.r[variation](f"{self.cache_dir}/RPipe.pssm"))


def fasta2pssm(df: pd.DataFrame,
               cache_path: str,
               database_path: str,
               varaiation: str,
               verbose: bool = False) -> dict:
    """

    :param df:
    :param cache_path:
    :param database_path:
    :param varaiation:
    :param verbose:
    :return:
    """
    # create the instance of PssmGetter
    pssm = PssmGetter(cache_dir=cache_path,
                      database_path=database_path,
                      verbose=verbose)
    # drop the duplicated UniProt_ID
    df.drop_duplicates(subset=["UniProt_ID"]).reset_index(drop=True, inplace=True)
    # fetch the reflection of {UniProt_ID : PSSM}
    return {row["UniProt_ID"]:
                pssm.get(ID=row["UniProt_ID"],
                         cData=row["Fasta"],
                         variation=varaiation)
            for rowIndex, row
            in df.iteritems()}


if __name__ == '__main__':
    # single test
    test_pssm = PssmGetter(cache_dir=r"D:/WorkPath/PycharmProjects/MutTm-pred/Dataset/Process4Dataset/feature_extractor/pssm/cache",
                           database_path=r"D:/WorkPath/PycharmProjects/MutTm-pred/Dataset/Process4Dataset/feature_extractor/pssm/UnirefDataset/uniref50.fasta",
                           verbose=True)
    test_pssm.get(ID="P11716",
                  cData=None,
                  variation="RPM_PSSM")
    # multiple test
    result = fasta2pssm()

    # pssm = PssmGetter(cache_path=r"D:/WorkPath/PycharmProjects/MutTm-pred/Dataset/Process4Dataset/feature_extractor/pssm/cache",
    #                   database_path=r"D:/WorkPath/PycharmProjects/MutTm-pred/Dataset/Process4Dataset/feature_extractor/pssm/UnirefDataset/uniref50.fasta",
    #                   verbose=True)
    # IDs = ["P11716", ]
    # sr = pd.DataFrame(IDs,
    #                   columns=["UniProt_ID"])
    # ans = PssmGotFromUrl(sr["UniProt_ID"],
    #                      variation="RPM_PSSM",
    #                      cache_path="./cache/pssm",
    #                      database_path=r"F:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Cache\tools\pssm\UnirefDataset\uniref50.fasta",
    #                      verbose=True)