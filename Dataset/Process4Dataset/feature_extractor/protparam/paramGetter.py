import json
import re
from io import StringIO
import numpy as np
import pandas as pd
import requests
import tqdm
import urllib3
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import os

urllib3.disable_warnings()

abs_path = os.path.dirname(__file__)  # 文件绝对路径
# abs_path = "D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Process4Dataset\feature_extractor\GOGetter.py"     # 当jupyter调用时可能无法使用__file__请自行修改
rel_path = os.path.relpath(abs_path).replace("\\", "/")  # 相对/调用路径
param_cache_path = os.path.join(rel_path, "protparam/param.npy").replace("\\", "/")


def single_uid(uniprot_id: str) -> tuple:
    fasta_response = requests.get(f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta?include=yes")
    if fasta_response.status_code == 200:
        # 提取序列部分（跳过标题行）
        fasta = fasta_response.text.strip()
        sequence = "".join(fasta.split('\n')[1:])
        return sequence, fasta
    else:
        raise Exception(f"{uniprot_id} do not have a param reflection in Uniprot.org. Please do it manuelly")


class ParamGetter:

    def __init__(self):
        # silent run
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_experimental_option("detach", True)
        # chrome
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.set_page_load_timeout(1800)
        # if chorme not work, try: https://googlechromelabs.github.io/chrome-for-testing/#stable download the fixed version to your chrome

    def params_got_from_url(self,
                            ID: str,
                            seqInfo: str) -> np.ndarray:
        """
        此处利用爬虫爬取ProtParam网站上有关蛋白质理化性质的特征，分别为:\n
        访问该文章无法使用代理\n
        1.Number of amino acids\n
        2.Molecular weight\n
        3.Theoretical pI\n
        4.Total number of atoms:\n
        5.Ext. coefficient\n
        6.The instability index\n
        7.Aliphatic index:\n
        注释掉的特征为MutTm-pred未选择的特征，有需要者可以取消注释.
        :param ID:
        :param seqInfo:
        :return:
        """
        # SeqGetter
        if pd.isna(seqInfo):
            # response = requests.post(f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta", verify=False)
            # fasta = StringIO("".join(response.text))
            # seqInfo = list(SeqIO.parse(fasta, 'fasta'))
            # seqInfo = str(seqInfo[0].sequence)
            # response.close()
            seqInfo = single_uid(ID)[0]

        # Searching Page
        self.driver.get("https://web.expasy.org/protparam/")  # 123
        seqPos = self.driver.find_element(By.NAME, "sequence")
        button = self.driver.find_element(By.CSS_SELECTOR,
                                          "#sib_body > form > p:nth-child(9) > input[type=submit]:nth-child(2)")
        seqPos.send_keys(seqInfo)
        button.click()
        html = self.driver.page_source

        # Result Page(handle with beautifulsoup)
        soup = BeautifulSoup(html, "html.parser")
        features = []
        coe_sum = 0
        coe_times = 1
        ext_existence = False  # check the existence of ext. coefficient
        for line in [item for item in soup.text.split("\n") if item != ""]:
            if len(line) == 1:
                continue
            if line.startswith("Number of amino acids:"):
                features.append(float(re.match(r"Number of amino acids:\s*([-\d.]+)", line).group(1)))
            elif line.startswith("Molecular weight:"):
                features.append(float(re.match(r"Molecular weight:\s*([-\d.]+)", line).group(1)))
            elif line.startswith("Theoretical pI:"):
                features.append(float(re.match(r"Theoretical pI:\s*([-\d.]+)", line).group(1)))
            # elif line.startswith("Total number of negatively charged residues (Asp + Glu):"):
            #     MachineLearning.append(float(re.match(r"Total number of negatively charged residues \(Asp \+ Glu\):\s*([-\d.]+)", line).group(1)))
            # elif line.startswith("Total number of positively charged residues (Arg + Lys):"):
            #     MachineLearning.append(float(re.match(r"Total number of positively charged residues \(Arg \+ Lys\):\s*([-\d.]+)", line).group(1)))
            elif line.startswith("Total number of atoms:"):
                features.append(float(re.match(r"Total number of atoms:\s*([-\d.]+)", line).group(1)))
            elif line.startswith("Ext. coefficient"):
                coe_sum += float(re.match(r"Ext. coefficient\s*([-\d.]+)", line).group(1))
                coe_times -= 1
                if not coe_times:
                    ext_existence = True
                    features.append(float(coe_sum))
            elif line.startswith("The instability index (II) is computed to be"):
                if not ext_existence:  # if ext. coefficient does not exist, we give it a NA then it will be filled with a average value after all calculations
                    features.append(1e9)
                features.append(
                    float(re.match(r"The instability index \(II\) is computed to be\s*([-\d.]+)", line).group(1)))
            elif line.startswith("Aliphatic index:"):
                features.append(float(re.match(r"Aliphatic index:\s*([-\d.]+)", line).group(1)))
            # elif line.startswith("Grand average of hydropathicity (GRAVY):"):
            #     MachineLearning.append(float(re.match(r"Grand average of hydropathicity \(GRAVY\):\s*([-\d.]+)", line).group(1)))

        if len(features) == 7:
            return np.array(features)
        # elif len(features) == 6:
        #     # features.insert(4, 1e9)
        #     print(f"Wrong Bug In Crawler on {uniprot_id}!And it is not the Ext. coefficient problem")
        #     return np.array(features)
        else:
            raise Exception(f"Crawler Error on {ID}!Please check it")

    def params_getter(self, df: pd.DataFrame) -> dict:
        # get the duplicated set
        df = df.drop_duplicates(subset=["UniProt_ID"]).reset_index(drop=True)

        # fill the empty column
        if "ProteinSeq" not in df.columns:
            df["ProteinSeq"] = pd.Series(index=df.index, data=pd.NA)

        # load the history file
        param_cache = np.load(param_cache_path, allow_pickle=True).item()

        # handle the data
        with tqdm.tqdm(total=df.shape[0]) as processingBar:
            result = dict()
            for rowIndex, row in df.iterrows():
                if row["UniProt_ID"] not in param_cache:
                    result[row["UniProt_ID"]] = self.params_got_from_url(row["UniProt_ID"], row["ProteinSeq"])
                    param_cache[row["UniProt_ID"]] = result[row["UniProt_ID"]]
                else:
                    result[row["UniProt_ID"]] = param_cache[row["UniProt_ID"]]
                processingBar.update(1)

        # save the history
        np.save(param_cache_path, np.array(param_cache))

        return result

    def __del__(self):
        self.driver.close()


if __name__ == '__main__':
    param = ParamGetter()
    td = pd.DataFrame(data=["P0AA04", "P02625", "P0A3H0", "P08821"],
                      columns=["UniProt_ID"])
    ans = param.params_getter(td)
