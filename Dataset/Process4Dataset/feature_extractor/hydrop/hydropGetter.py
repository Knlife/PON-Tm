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
from selenium.webdriver.support.select import Select
from bs4 import BeautifulSoup
import os

urllib3.disable_warnings()

abs_path = os.path.dirname(__file__)  # 文件绝对路径
# abs_path = "D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Process4Dataset\feature_extractor\GOGetter.py"     # 当jupyter调用时可能无法使用__file__请自行修改
rel_path = os.path.relpath(abs_path).replace("\\", "/")  # 相对/调用路径
scale_cache_path = os.path.join(rel_path, "hydrop/scale.npy").replace("\\", "/")


def single_uid(uniprot_id: str) -> tuple:
    fasta_response = requests.get(f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta?include=yes")
    if fasta_response.status_code == 200:
        # 提取序列部分（跳过标题行）
        fasta = fasta_response.text.strip()
        sequence = "".join(fasta.split('\n')[1:])
        return sequence, fasta
    else:
        raise Exception(f"{uniprot_id} do not have a param reflection in Uniprot.org. Please do it manuelly")


class ScaleGetter:

    def __init__(self):
        # silent run
        self.chrome_options = webdriver.ChromeOptions()
        # self.chrome_options.add_argument('--headless')
        # self.chrome_options.add_experimental_option("detach", True)
        # chrome
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver_demo = webdriver.Chrome(options=self.chrome_options)
        self.driver.set_page_load_timeout(1800)
        self.driver_demo.set_page_load_timeout(1800)
        # if chorme not work, try: https://googlechromelabs.github.io/chrome-for-testing/#stable download the fixed version to your chrome

    def scale_got_from_url(self,
                           ID: str,
                           target_index: int,
                           seqInfo: str) -> np.ndarray:
        """
        此处利用爬虫爬取ProtParam网站上有关蛋白质理化性质的特征，分别为:\n
        访问该文章无法使用代理\n
        注释掉的特征为Pon-DT未选择的特征，有需要者可以取消注释.
        :param target_index:
        :param ID:
        :param seqInfo:
        :return:
        """
        # get the protein sequence from uniprot
        if pd.isna(seqInfo):
            seqInfo = single_uid(ID)[0]

        # 可选的57种标注方式
        target_types = ["Molecular weight", "Number of codon(s)", "Bulkiness", "Polarity / Zimmerman",
                        "Polarity / Grantham",
                        "Refractivity", "Recognition factors", "Hphob. / Eisenberg et al.", "Hphob. OMH / Sweet et al.",
                        "Hphob. / Hopp &amp; Woods", "Hydropath. / Kyte &amp; Doolittle", "Hphob. / Manavalan et al.",
                        "Hphob. / Abraham &amp; Leo", "Hphob. / Black", "Hphob. / Bull &amp; Breese",
                        "Hphob. / Fauchere et al.", "Hphob. / Guy", "Hphob. / Janin", "Hphob. / Miyazawa et al.",
                        "Hphob. / Rao &amp; Argos", "Hphob. / Roseman", "Hphob. / Tanford", "Hphob. / Wolfenden et al.",
                        "Hphob. / Welling &amp; al", "Hphob. HPLC / Wilson &amp; al", "Hphob. HPLC / Parker &amp; al",
                        "Hphob. HPLC pH3.4 / Cowan", "Hphob. HPLC pH7.5 / Cowan", "Hphob. / Rf mobility",
                        "HPLC / HFBA retention", "HPLC / TFA retention", "Transmembrane tendency",
                        "HPLC / retention pH 2.1",
                        "HPLC / retention pH 7.4", "% buried residues", "% accessible residues", "Hphob. / Chothia",
                        "Hphob. / Rose &amp; al", "Ratio hetero end/side", "Average area buried", "Average flexibility",
                        "alpha-helix / Chou &amp; Fasman", "beta-sheet / Chou &amp; Fasman",
                        "beta-turn / Chou &amp; Fasman",
                        "alpha-helix / Deleage &amp; Roux", "beta-sheet / Deleage &amp; Roux",
                        "beta-turn / Deleage &amp; Roux", "Coil / Deleage &amp; Roux", "alpha-helix / Levitt",
                        "beta-sheet / Levitt", "beta-turn / Levitt", "Total beta-strand", "Antiparallel beta-strand",
                        "Parallel beta-strand", "A.A. comp. in Swiss-Prot", "Relative mutability"]

        # 答案存储列表
        scale_results = []

        for selected_type in target_types:
            # 进入主页面
            self.driver.get("https://web.expasy.org/protscale/")
            # 传入序列信息
            sequence_input = self.driver.find_element(By.NAME, "sequence")
            sequence_input.send_keys(seqInfo)
            # 选择窗口大小
            window_size_input = Select(self.driver.find_element(By.CSS_SELECTOR,
                                                                "#sib_body > form > select"))
            window_size_input.select_by_visible_text("3")
            # 选择标注类型
            type_input_apis = self.driver.find_elements(By.NAME, "scale")
            for api in type_input_apis:
                if api.get_attribute("value") == selected_type:
                    api.click()
                    break
            else:
                raise Exception(f"There is not such type in ProtScale, {selected_type}")
            # 点击提交按钮
            submit_button = self.driver.find_element(By.CSS_SELECTOR,
                                                     "#sib_body > form > p:nth-child(22) > input[type=submit]:nth-child(1)")
            submit_button.click()

            # 点击获取简洁数据
            href = self.driver.find_element(By.CSS_SELECTOR, "#sib_body > ul > li:nth-child(4) > a")
            href.click()

            # 读取简洁数据并存储
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            for row in soup.text.split("\n"):
                index, score = map(lambda num: float(num) if "." in num else int(num), row.strip().split())
                if target_index == index:
                    scale_results.append(score)

        return np.array(scale_results)

    def scale_getter(self, df: pd.DataFrame) -> dict:
        # get the duplicated set
        df = df.drop_duplicates(subset=["UniProt_ID"]).reset_index(drop=True)

        # fill the empty column
        if "ProteinSeq" not in df.columns:
            df["ProteinSeq"] = pd.Series(index=df.index, data=pd.NA)

        # load the history file
        scale_cache = np.load(scale_cache_path, allow_pickle=True).item()

        # handle the data
        with tqdm.tqdm(total=df.shape[0]) as processingBar:
            result: dict = dict()
            for rowIndex, row in df.iterrows():
                # basic info
                ID = row["UniProt_ID"]
                index = row["MutationIndex"]
                wild_residue = row["AminoAcidFrom"]
                mutant_residue = row["AminoAcidTo"]
                wild_sequence = row["ProteinSeq"]
                mutant_sequence = wild_sequence[: index - 1] + mutant_residue + wild_sequence[index:]
                # dict label
                wild_label = "-".join([ID, str(index), wild_residue])
                mutant_label = "-".join([ID, str(index), mutant_residue])
                # wild_embedding scores
                if wild_label not in scale_cache:
                    result[wild_label] = self.scale_got_from_url(ID, index, wild_sequence)
                    scale_cache[wild_label] = result[wild_label]
                else:
                    result[wild_label] = scale_cache[wild_label]
                # mutant scores
                if mutant_label not in scale_cache:
                    result[mutant_label] = self.scale_got_from_url(ID, index, mutant_sequence)
                    scale_cache[mutant_label] = result[mutant_label]
                else:
                    result[mutant_label] = scale_cache[mutant_label]
                processingBar.update(1)

        # save the history
        np.save(scale_cache_path, np.array(scale_cache))

        # return teh result
        return result

    def __del__(self):
        self.driver.close()


if __name__ == '__main__':
    param = ScaleGetter()
    td = pd.DataFrame(data=["P0AA04", "P02625", "P0A3H0", "P08821"],
                      columns=["UniProt_ID"])
    ans = param.scale_getter(td)

    # np.save(scale_cache_path, np.array({"None": [0, 0, 0, 0, 0, 0, 0]}))
