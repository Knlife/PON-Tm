import collections
import json
import os.path
from Bio import ExPASy
from Bio import SwissProt
import pandas as pd
abs_path = os.path.dirname(os.path.abspath(__file__))
rel_path = os.path.relpath(abs_path, os.getcwd())


# csv file download from https://ftp.ebi.ac.uk/pub/databases/msd/sifts/csv/pdb_chain_uniprot.csv
def offlineGenerator() -> collections.defaultdict:
    import json
    if not os.path.exists("pdb_chain_uniprot.csv"):
        import requests
        ref_file = requests.get("https://ftp.ebi.ac.uk/pub/databases/msd/sifts/csv/pdb_chain_uniprot.csv", stream=True)
        with open("pdb_chain_uniprot.csv", "wb") as csv_file:
            for chunk in ref_file.iter_content(chunk_size=1024):
                if chunk:
                    csv_file.write(chunk)
    # 删除可对应多个UniPortID的条目，并构建字典存储在缓存文件中
    print("正在重新生成缓存文件pdb2uid.json...")
    offline = pd.read_csv("pdb_chain_uniprot.csv")[["PDB", "SP_PRIMARY"]]
    offline.drop_duplicates(subset=["PDB", "SP_PRIMARY"], keep=False, inplace=True)
    offline.reset_index(drop=True, inplace=True)

    # 转换为映射字典
    pdb2uid_ref = collections.defaultdict(str)
    for rowIndex, row in offline.iterrows():
        pdb2uid_ref[row["PDB"]] = row["SP_PRIMARY"]

    # 存储为json文件
    with open("pdb2uid.json", "w") as json_file:
        json.dump(pdb2uid_ref, json_file)

    return pdb2uid_ref


if not os.path.exists("pdb2uid.json"):
    offlineGenerator()
pdb2uid_ref = json.load(open("pdb2uid.json"))


def onlineGetter(pdb_id):
    handle = ExPASy.get_sprot_raw(pdb_id)
    record = SwissProt.read(handle)
    return record.entry_name


def single_getter(pdb_id):
    global pdb2uid_ref
    try:
        return onlineGetter(pdb_id)
    except:
        res = pdb2uid_ref.get(pdb_id, None)
        if res is None:
            return "None"
        else:
            return res


def multi_getter(pdb_ids):
    return {pdb_id: single_getter(pdb_id)
            for pdb_id in pdb_ids}


# 本地测试
if __name__ == "__main__":
    result_single = single_getter("1aky")
    result_mult = multi_getter(["1aky", "1akx"])
