import os

import requests
import urllib3
import numpy as np
import pandas as pd
import tqdm
import subprocess

# R_HOME在Windows上设置有问题，必须在此处进行将其添加入环境变量才能调用rpy2
os.environ['R_HOME'] = r"C:\Program Files\R\R-4.3.2"
from rpy2 import robjects
robjects.r("""
library(PSSMCOOL)
ab_pssm <- function (pssm_path){
    return(AB_PSSM(pssm_path))
}
rpm_pssm <- function (pssm_path){
    return(RPM_PSSM(pssm_path))
}
ab_pssm <- function (pssm_path){
    return(AB_PSSM(pssm_path))
}
ab_pssm <- function (pssm_path){
    return(AB_PSSM(pssm_path))
}
""")

# 增加POST首部已避免Uniprot的反爬，并且阻止urllib3的报错以避免错误信息
header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
urllib3.disable_warnings()
# 路径问题
rel_path = os.path.relpath(os.path.dirname(__file__))
cache_path = os.path.join(rel_path, r"pssm/cache")
default_database_path = os.path.join(rel_path, r"pssm/UnirefDataset/uniref50.fasta")
# default_database_path = os.path.join(rel_path, r"pssm/SwissProtDataset/swissprot")


def PssmGotFromUrl(df: pd.DataFrame,
                   variation: str,
                   database_path: str = None,
                   mode: str = "PSSMCOOL",
                   verbose: bool = False) -> dict:
    """
    全自动化：利用UniProt_ID生成.fasta文件\n
    再利用该.fasta文件本地生成PSSM比对矩阵\n
    最后利用R包处理成对应变式矩阵
    :param df: 待检测的DataFrame，其中必须包含UniProt信息
    :param cache_path: 缓存路径 由于该函数调用了一R函数并产生输出文件，故需要设置一缓存文件以满足
    :param database_path: 用以蛋白质序列比对(Blast)的数据库路径，保证该数据库以及Blast初始化
    :param variation: 所选的PSSM变种，见 https://github.com/BioCool-Lab/PSSMCOOL
    :param mode: 采用的PSSM-Variant获取工具，前采用PSSMPro存在效率问题，现用PSSMCOOL
    :param verbose: 是否打印信息
    :return: {ID1 : PSSM_Variant1, ...IDN : PSSM_VariantN}
    """
    global cache_path

    def variation_getter_from_pssmcool(ID: str) -> np.ndarray:
        """
        利用R包PSSMCOOL生成PSSM变式矩阵的函数
        :return:
        """
        # R in windows never use "\\", but os.getcwd() return a path with "\\"
        ID_pssm_cache_path = cache_path + f"/pssm/{ID}.pssm"
        if not os.path.exists(ID_pssm_cache_path):
            print(ID_pssm_cache_path)
            raise Exception("Path error in pssmGetter, check")
        var_pssm = list(robjects.r[variation](ID_pssm_cache_path))
        return np.array(var_pssm)

    def pssm_getter_from_blast(ID: str,
                               fasta: str) -> np.ndarray:
        if not os.path.exists(cache_path + f"/pssm/{ID}.pssm"):
            if pd.isna(fasta):
                url = f"https://www.uniprot.org/uniprot/{ID}.fasta"
                response = requests.post(url, headers=header, verify=False)
                fasta = "".join(response.text)

            # 写入Fasta文件
            with open(fr"{cache_path}/fasta_file.txt", "w") as basic:
                fasta = [item + "\n" for item in fasta.split("\n") if not item.isspace() and item]
                basic.writelines(fasta)
            # 运行psi-blast
            command = "psiblast {} {} {} {} {} {} {}".format("-comp_based_stats " + "1",
                                                             "-db " + default_database_path,
                                                             "-query " + cache_path + "/fasta_file.txt",
                                                             "-evalue " + "0.001",
                                                             "-num_iterations " + "3",
                                                             "-num_threads " + "20",
                                                             "-out_ascii_pssm " + cache_path + f"/pssm/{ID}.pssm")
            process = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
            if process.returncode:
                raise Exception(f"Psiblast failed in {ID}")

        # 根据原PSSM矩阵求取变式
        if mode == "PSSMCOOL":
            return variation_getter_from_pssmcool(ID)
        else:
            raise Exception("Wrong mode has been selected")

    # check if the fasta exists
    if "Fasta" not in df.columns:
        df["Fasta"] = df["UniProt_ID"].apply(lambda x: pd.NA)

    # get the duplicated set
    df = df[["UniProt_ID", "Fasta"]].drop_duplicates(subset=["UniProt_ID"]).reset_index(drop=True)

    # check if the file exists
    if database_path is not None:
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Database not found: {database_path}")
        else:
            global default_database_path
            default_database_path = database_path

    # generate the reflection dict
    with tqdm.tqdm(total=df.shape[0]) as processingBar:
        result = dict()
        for rowIndex, row in df.iterrows():
            result[row["UniProt_ID"]] = pssm_getter_from_blast(row["UniProt_ID"], row["Fasta"])
            processingBar.update(1)

        return result


if __name__ == "__main__":
    IDs = ["Q8GB52", "P09372"]
    td = pd.DataFrame(IDs,
                      columns=["UniProt_ID"])
    rr = PssmGotFromUrl(td, "RPM_PSSM")
