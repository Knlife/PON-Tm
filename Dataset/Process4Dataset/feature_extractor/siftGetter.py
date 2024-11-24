import os.path
from os import remove
from os.path import dirname, join, exists
import subprocess
import pandas as pd
import tqdm


rel_path = os.path.relpath(dirname(__file__))
sub_dir = join(rel_path, "sift4g/release/mutations")                        # 变异文件存储路径
out_dir = join(rel_path, "sift4g/release/results")                          # 结果文件存储路径
executor_path = join(rel_path, "sift4g/bin/sift4g")                         # sift4g执行路径
query_path = join(rel_path, "sift4g/release/query.fasta")                   # 序列查询文件路径
default_database_path = join(rel_path, "sift4g/release/uniref50.fasta")     # 数据库文件路径 Default:UniRef50
wronglog_path = join(rel_path, "sift4g/release/wrong.txt")                  # 错误日志


def check_necessity(sub_path: str,
                    pred_path: str,
                    mutations: list):
    if not exists(pred_path):
        return True, mutations

    add_mutations = []
    with open(sub_path, "r") as mutation_file:
        pre_mutations = set([m.rstrip("\n") for m in mutation_file.readlines()])
        for mutation in mutations:  # 遍历传入的突变是否在之前突变之中
            if mutation not in pre_mutations:
                add_mutations.append(mutation)

    if len(add_mutations) != 0:
        print("新增突变：", add_mutations)
        remove(pred_path)
        return True, list(pre_mutations) + add_mutations
    else:
        return False, mutations


def single_getter(ID: str,
                  fasta: str,
                  mutations: list):

    # region 输出运行日志并设置运行路径
    print(ID, len(mutations), end=" ")
    sub_path = f"{sub_dir}/{ID}.subst".replace("\\", "/")
    pred_path = f"{out_dir}/{ID}.SIFTprediction".replace("\\", "/")
    # endregion

    # region 检查存储的变异信息是否满足当前需求的变异，若不满足或没有处理过则重新处理
    flag, mutations = check_necessity(sub_path, pred_path, mutations)
    if flag:
        # 写入.fasta文件
        with open(query_path, "w") as out:
            _tmp = [item for item in fasta.split("\n")]
            _tmp[0] = ">" + ID
            out.write("\n".join(_tmp))

        # 写入.subst文件
        with open(sub_path, "w") as mutation_file:
            mutation_file.writelines([mutation + "\n" for mutation in mutations[:-1]] + [mutations[-1]])

        # 执行Sift4G
        command = f"wsl {executor_path} -q {query_path} --subst {sub_dir} --out {out_dir} -d {default_database_path} -t 16".replace("\\", "/")  # wsl只能使用相对路径
        result = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 若SIFT4G执行出现错误，将错误ID置入错误日志并删除可能存在的SIFT4G-result文件
        if result.returncode:
            # if exists(pred_path):
            #     remove(pred_path)
            # if exists(sub_path):
            #     remove(sub_path)
            print(f"===Exception raised in {ID}, Check the wrong.txt===")
            with open(wronglog_path, "a") as f:
                f.write(ID + mutations + "\n")
            return {}
    # endregion

    # region 读取结果文件并处理
    with open(pred_path) as out:
        sift_scores = {}
        text = out.readlines()
        for item in text:
            if item.startswith("WA"):
                continue
            tmp_item = item.split()
            sift_scores[ID + "-" + tmp_item[0]] = list(map(float, tmp_item[2:]))
    # endregion

    # region 检查是否存在重复Sift4GScore，若程序很可能出现问题
    # global pre_scores
    # if sift_scores == pre_scores:
    #     print("Sift4g meet the same result in ", uniprot_id)
    # else:
    #     pre_scores = sift_scores
    # 返回处理结果
    # endregion

    # print(sift_scores)
    return sift_scores


def mult_getter(df: pd.DataFrame,
                dataset_version: str,
                database_path: str = None,
                ) -> dict:
    """
    利用SIFT4G对氨基酸突变进行打分
    :param dataset_version:
    :param database_path:
    :param sift4g_path:
    :param df: 包含["UniPort_ID", "Mutation"]的Dataframe
    :return: {"-".join("UniPort_ID", "Mutation"): [sift4g_socre1, sift4g_score2]}
    """
    # 初始化出映射列表，利用双指针进行处理
    # def reflection_generator():
    #     sift4g_ans = {}
    #     label_step = 0                      # 进度指针
    #     label_ID = df.at[0, "UniProt_ID"]   # ID指针
    #     length_td = len(df)
    #     mutations = set()  # 临时存储替换的列表
    #
    #     with tqdm.tqdm(total=length_td) as processingBar:
    #         while True:
    #
    #             # 如果没有越界并且当前ID发生变化
    #             if label_step < length_td and df.at[label_step, "UniProt_ID"] != label_ID:
    #                 # print("Acting on ", label_ID, mutations)
    #                 tmp_ans = single_getter(label_ID, df.at[label_step - 1, "Fasta"], list(mutations))
    #                 processingBar.update(len(mutations))    # 更新进度条
    #                 sift4g_ans.update(tmp_ans)
    #                 mutations.clear()   # 清空突变列表
    #                 label_ID = df.at[label_step, "UniProt_ID"]  # 更换当前ID指针
    #
    #             # 如果是越界
    #             if label_step == length_td:
    #                 sift4g_ans.update(single_getter(label_ID, df.at[label_step - 1, "Fasta"], list(mutations)))
    #                 processingBar.update(len(mutations))    # 更新进度条
    #                 break
    #
    #             # 如果并不是越界，则是一般情况，直接前移
    #             mutations.add(df.at[label_step, "Mutation"])
    #             label_step += 1
    #
    #     return sift4g_ans

    # select the columns

    df = df[["UniProt_ID", "Mutation", "Fasta"]]
    df = df.drop_duplicates(subset=["UniProt_ID", "Mutation"], keep="first").reset_index(drop=True)
    df = df.sort_values(by=["UniProt_ID", "Mutation"], ascending=True).reset_index(drop=True)

    # check if the database exists
    if database_path is not None:
        if not exists(database_path):
            raise FileNotFoundError(f"Database not found: {database_path}")
        else:
            global default_database_path
            default_database_path = database_path

    # generate the relection dict
    reflection = {}
    label_step = 0                          # 进度指针
    label_ID = df.at[0, "UniProt_ID"]       # ID指针
    mutations = set()                       # 临时存储替换的列表

    with tqdm.tqdm(total=len(df)) as processingBar:
        while True:
            # 如果下标发生越界
            if label_step >= len(df):
                reflection.update(single_getter(label_ID,
                                                df.at[label_step - 1, "Fasta"],
                                                list(mutations)))
                break

            # 如果UniProtID发生变化，则需要对前一ID-Mutations求取Sift4GScores，并更换当前ID
            if df.at[label_step, "UniProt_ID"] != label_ID:
                reflection.update(single_getter(label_ID,
                                                df.at[label_step - 1, "Fasta"],
                                                list(mutations)))
                mutations.clear()  # 清空突变列表
                label_ID = df.at[label_step, "UniProt_ID"]  # 修改当前ID指针

            # 如果UniProtID没有变化，则可以直接添加当前变异信息并前移指针
            mutations.add(df.at[label_step, "Mutation"])
            processingBar.update(1)
            label_step += 1

    # utilize reflection to get the scores
    return {row["UniProt_ID"] + "-" + row["Mutation"]: reflection[row["UniProt_ID"] + "-" + row["Mutation"]]
            for index, row
            in df.iterrows()}
