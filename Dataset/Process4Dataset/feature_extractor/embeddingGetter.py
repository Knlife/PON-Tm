import numpy as np
import pandas as pd
import torch
from transformers.testing_utils import torch_device


class DeepEncoder():
    """
    用于提取深度学习蛋白质大语言模型
    """

    def __init__(self,
                 embedding_model_path,
                 embedding_model_name,
                 device,
                 batch: int = 32):
        self.embedding_model_path = embedding_model_path
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.batch = batch
        self.embedding_tokenizer, self.embedding_model = self.embedding_model_getter()
        print("嵌入模型后显存使用量: ", torch.cuda.memory_allocated(device) / (1024 ** 3))

    def embedding_model_getter(self):
        if self.embedding_model_name in {"ESM-2-150M", "ESM-2-650M", "ESM-2-3B"}:
            from transformers import EsmTokenizer, EsmModel
            embedding_tokenizer = EsmTokenizer.from_pretrained(self.embedding_model_path)
            embedding_model = EsmModel.from_pretrained(self.embedding_model_path).to(self.device)

        elif self.embedding_model_name in {"ProtT5-Half", }:
            from transformers import T5Tokenizer, T5EncoderModel
            embedding_tokenizer = T5Tokenizer.from_pretrained(self.embedding_model_path, do_lower_case=False)
            embedding_model = T5EncoderModel.from_pretrained(self.embedding_model_path).to(self.device)

        elif self.embedding_model_name in {"ProtBert", }:
            from transformers import BertTokenizer, BertModel
            embedding_tokenizer = BertTokenizer.from_pretrained(self.embedding_model_path, do_lower_case=False,
                                                                legacy=True)
            embedding_model = BertModel.from_pretrained(self.embedding_model_path).to(self.device)

        else:
            from transformers import AutoTokenizer, AutoModel
            try:
                embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)
                embedding_model = AutoModel.from_pretrained(self.embedding_model_path).to(self.device)
            except Exception:
                raise ValueError(
                    f"Can not load the model from {self.embedding_model_path}, please check in huggingface.co")

        return embedding_tokenizer, embedding_model

    def encode(self, sequence: str) -> np.ndarray:
        if self.embedding_model_name in {"ProtBert", "ProtT5-Half"}:
            sequence = " ".join(sequence)
        token = self.embedding_tokenizer(" ".join(sequence), return_tensors="pt", padding=False).to(self.device)
        with torch.no_grad():
            embedding = self.embedding_model(**token).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        return embedding

    def encode_mult(self,
                    ids: list,
                    sequences: list) -> dict:
        # check the sequences format
        if self.embedding_model_name in {"ProtBert", "ProtT5-Half"}:
            sequences = [" ".join(sequence) for sequence in sequences]
        # tokenize sequences
        tokens = self.embedding_tokenizer.batch_encode_plus(sequences, add_special_tokens=True)
        input_ids = torch.tensor(tokens['input_ids']).to(self.device)
        attention_masks = torch.tensor(tokens['attention_mask']).to(self.device)
        # generate embeddings
        with torch.no_grad():
            embeddings = self.embedding_model(input_ids, attention_masks)
        # generate the uniprot_id->embedding mapping
        return {ids[pos]: embeddings.last_hidden_state[pos].mean(dim=0).cpu().numpy() for pos in range(len(ids))}

    def context_encode(self, sequence: str) -> np.ndarray:
        if self.embedding_model_name in {"ProtBert", "ProtT5-Half"}:
            sequence = " ".join(sequence)
        token = self.embedding_tokenizer(sequence, return_tensors="pt", padding=False).to(self.device)
        with torch.no_grad():
            embedding = self.embedding_model(**token).last_hidden_state[:, 1: -1, :].mean(dim=2).squeeze().cpu().numpy()

        embedding_max = np.max(embedding, axis=0)
        embedding_min = np.min(embedding, axis=0)
        embedding_range = embedding_max - embedding_min
        embedding = (embedding - embedding_min) / embedding_range

        return embedding

    def encode_from_list(self,
                         ids: list,
                         sequences: list) -> dict:
        res = {}
        for beg in range(0, len(sequences), self.batch):  # Consider the GPU memory, 32 is the batch size
            beg, end = beg, min(beg + self.batch, len(sequences))
            res.update(self.encode_mult(ids[beg: beg + self.batch], sequences[beg: beg + self.batch]))
        return res

    def encode_from_list_seperated(self,
                                   ids: list,
                                   sequences: list) -> dict:
        return {ID: self.encode(sequence) for ID, sequence in zip(ids, sequences)}

    def context_encode_from_list_seperated(self,
                                           ids: list,
                                           sequences: list) -> dict:
        return {ID: self.context_encode(sequence) for ID, sequence in zip(ids, sequences)}

    def encode_from_df(self,
                       df: pd.DataFrame) -> dict:
        if "WildSeq" in df.columns:
            df = df.drop_duplicates(subset=["UniProt_ID", "WildSeq"], keep="first")
            return self.encode_from_list(df["UniProt_ID"].to_list(), df["WildSeq"].to_list())
        elif "MutantSeq" in df.columns:
            df = df.drop_duplicates(subset=["UniProt_ID", "MutantSeq"], keep="first")
            return self.encode_from_list(df["UniProt_ID"].to_list(), df["MutantSeq"].to_list())
        elif "WildCutSeq" in df.columns:
            df = df.drop_duplicates(subset=["UniProt_ID", "WildCutSeq"], keep="first")
            return self.encode_from_list(df["UniProt_ID"].to_list(), df["WildCutSeq"].to_list())
        elif "MutantCutSeq" in df.columns:
            df = df.drop_duplicates(subset=["UniProt_ID", "MutantCutSeq"], keep="first")
            return self.encode_from_list(df["UniProt_ID"].to_list(), df["MutantCutSeq"].to_list())
        else:
            raise ValueError(
                "The dataframe should contain at least one of WildSeq, MutantSeq, WildCutSeq, MutantCutSeq")

    def encode_from_df_seperated(self,
                                 df: pd.DataFrame) -> dict:
        if "WildSeq" in df.columns:
            df = df.drop_duplicates(subset=["UniProt_ID", "WildSeq"], keep="first")
            return self.encode_from_list_seperated(df["UniProt_ID"].to_list(), df["WildSeq"].to_list())
        elif "MutantSeq" in df.columns:
            df = df.drop_duplicates(subset=["UniProt_ID", "MutantSeq"], keep="first")
            return self.encode_from_list_seperated(df["UniProt_ID"].to_list(), df["MutantSeq"].to_list())
        elif "WildCutSeq" in df.columns:
            df = df.drop_duplicates(subset=["UniProt_ID", "WildCutSeq"], keep="first")
            return self.context_encode_from_list_seperated(df["UniProt_ID"].to_list(), df["WildCutSeq"].to_list())
        elif "MutantCutSeq" in df.columns:
            df = df.drop_duplicates(subset=["UniProt_ID", "MutantCutSeq"], keep="first")
            return self.context_encode_from_list_seperated(df["UniProt_ID"].to_list(), df["MutantCutSeq"].to_list())
        else:
            raise ValueError(
                "The dataframe should contain at least one of WildSeq, MutantSeq, WildCutSeq, MutantCutSeq")

    def release(self):
        self.embedding_model = self.embedding_model.to(torch.device("cpu"))
