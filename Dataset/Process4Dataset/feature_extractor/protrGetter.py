# -*- coding : UTF-8 -*-
# @Author : JieJue
# @Project : MutTm-pred
# @Time : 2023/11/14 22:06
# @Name : protr
# @IDE : PyCharm
# @Introduction : as follows
"""

"""
import os
import numpy as np
import pandas as pd
import requests
import tqdm
import urllib3
# R_HOME在Windows上设置有问题，必须在此处进行设置后才能调用rpy2
os.environ['R_HOME'] = r"C:\Program Files\R\R-4.3.2"
from rpy2 import robjects
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
urllib3.disable_warnings()
robjects.r("""
protr <- function (cache_file){
    library(protr)
    selected <- readFASTA(paste0(cache_file, "/fasta_file.fasta"))
    # Common descriptors
    # ACC -- 20
    acc <- t(sapply(selected, extractAAC))
    # DC -- 400
    dc <- t(sapply(selected, extractDC))
    # TC -- 8000 -D
    tc <- t(sapply(selected, extractTC))
    # MoreauBroto -- 240 -D
    moreau <- t(sapply(selected, extractMoreauBroto))
    # Moran -- 240 -D
    moran <- t(sapply(selected, extractMoran))
    # Geary -- 240 -D
    geary <- t(sapply(selected, extractGeary))
    # CTDC -- 21
    ctdc <- t(sapply(selected, extractCTDC))
    # CTDT -- 21
    ctdt <- t(sapply(selected, extractCTDT))
    # CTDD -- 105
    ctdd <- t(sapply(selected, extractCTDD))
    # CTtiad -- 343 -D
    ctriad <- t(sapply(selected, extractCTriad))
    # SOCN -- 60 -D
    socn <- t(sapply(selected, extractSOCN))
    # QSO -- 100
    qso <- t(sapply(selected, extractQSO))
    # PAAC -- 50
    paac <- t(sapply(selected, extractPAAC))
    # APAAC -- 80
    apaac <- t(sapply(selected, extractAPAAC))
    # Whole -- 797
    # whole <- c(acc, dc, tc, moreau, moran, geary, ctdc, ctdt, ctdd, ctriad, socn, qso, paac, apaac)
    return(list(ACC=acc, DC=dc, TC=tc, MOREAU=moreau, MORAN=moran, GEARY=geary, CTDC=ctdc,
                CTDT=ctdt, CTDD=ctdd, CTRIAD=ctriad, SOCN=socn, QSO=qso, PAAC=paac, APAAC=apaac))
}
""")
rel_path = os.path.relpath(os.path.dirname(__file__)).replace("\\", "/")    # 获取调用文件之于该文件的相对路径
abs_path = os.path.dirname(__file__).replace("\\", "/")                     # 获取当前文件的绝对路径
working_path = os.path.join(abs_path, "protr").replace("\\", "/")           # 获取protr软件包的缓存路径
pre_scores = [0 for _ in range(10)]


def mult_getter(df: pd.DataFrame,
                protr_list: list = None,
                verbose: bool = True) -> dict:
    """
    if input dataframe does not contain "Fasta", just leave it empty or pd.NA.
    :param df:
    :param protr_list:
    :param verbose:
    :return:
    """
    global working_path

    def ProtrGetterFromIO(ID: str,
                          fasta: str = None) -> np.ndarray:
        global pre_scores
        # if fasta does not exist in input, get it from uniprot
        if pd.isna(fasta):
            url = f"https://www.uniprot.org/uniprot/{ID}.fasta"
            response = requests.post(url, headers=header, verify=False)
            fasta = "".join(response.text)

        # write the Fasta into the file
        with open(fr"{working_path}/fasta_file.fasta", "w") as basic:
            fasta = [item for item in fasta.split("\n") if not item.isspace() and item]
            for item in fasta:
                basic.write(item + "\n")

        # R in windows never use "\\", but os.getcwd() return a path with "\\"
        mid = robjects.r["protr"](working_path)
        ans = []
        for feature in protr_list:
            ans.extend(list(mid.rx2(feature)))

        # check if the io is right
        if ans[:10] != pre_scores:
            pre_scores = ans[:10]
        else:
            print("io error in {protr}")

        return np.array(ans)

    # get the features list
    if not protr_list:
        protr_list = ["ACC", "DC", "TC", "MOREAU", "MORAN", "GEARY", "CTDC", "CTDT",
                      "CTDD", "CTRIAD", "SOCN", "QSO","PAAC", "APAAC"]

    # fill the empty column
    if "Fasta" not in df.columns:
        df["Fasta"] = df["UniProt_ID"].apply(lambda x: pd.NA)

    # get the duplicated set
    df = df[["UniProt_ID", "Fasta"]].drop_duplicates(subset=["UniProt_ID"]).reset_index(drop=True)

    # generate the reflection dict
    with tqdm.tqdm(total=df.shape[0]) as pbar:
        result = dict()
        for rowIndex, row in df.iterrows():
            result[row["UniProt_ID"]] = ProtrGetterFromIO(row["UniProt_ID"], row["Fasta"])
            pbar.update(1)
        return result


if __name__ == '__main__':
    IDs = ["Q8GB52", "P09372"]
    td = pd.DataFrame(IDs,
                      columns=["UniProt_ID"])
    rr = mult_getter(td)
