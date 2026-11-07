import os
import pickle
import random
from typing import Iterator, Sequence

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, Sampler

tokenizer = Tokenizer.from_file("data/wmt14_en_de/bpe37k_shared.json")
from tqdm import tqdm
import pickle
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from array import array

class MyDataSet(Dataset):
    def __init__(self, src_file, tgt_file,valid_ids_cache = None,mode ="train"):
        self.src_file, self.tgt_file = src_file, tgt_file
        self.src_off = self._offsets(src_file)
        self.tgt_off = self._offsets(tgt_file)
        self.len = min(len(self.src_off), len(self.tgt_off)) - 1
        self.max_seq_len = 4000
        self._fh = {}
        if valid_ids_cache is not None and  os.path.exists(valid_ids_cache):
            with open(valid_ids_cache, "rb") as f:
                self.valid_ids = pickle.load(f)
            print("从缓存中加载有效句子ID,总共：" + str(len(self.valid_ids)) +"条")
            self.len = len(self.valid_ids.keys())

        else:
            if mode == "train":
                print("生成有效句子ID缓存")
                self.valid_ids = {}
                # ---------------------------
                # 1.去除太长的句子防止爆显存
                # ---------------------------
                t = tqdm(range(self.len))
                for i in range(self.len) :
                    src = tokenizer.encode(self._read(self.src_file, self.src_off, i))
                    tgt = tokenizer.encode(self._read(self.tgt_file, self.tgt_off, i))
                    if len(src.tokens) >self.max_seq_len  or len(tgt.tokens) > self.max_seq_len:
                        continue
                    self.valid_ids[i] = len(src.tokens) + len(tgt.tokens)
                    t.update(1)
                t.close()
                # ----------------------------
                # 2. 按照句子长度排序
                # ----------------------------
                self.valid_ids = dict(sorted(self.valid_ids.items(), key=lambda x: x[1]))
                self.len = len(self.valid_ids)
                # ----------------------------
                # 3.保存有效句子ID缓存
                # ----------------------------
                with open("valid_ids_cache.pkl", "wb") as f:
                    pickle.dump(self.valid_ids, f)
                print("有效句子ID缓存保存完成"+"长度为"+str(self.len))

        self._fh = {}

    def get_valid(self):
        return list(self.valid_ids.keys())


    @staticmethod
    def _offsets(path):
        offs = array("q", [0])          # 每行起始字节位置
        pos = 0
        with open(path, "rb") as f:      # 二进制模式，不需要解码
            for line in f:
                pos += len(line)
                offs.append(pos)
        return offs

    def _read(self, path, offs, i):
        f = self._fh.get(path)
        if f is None:
            f = open(path, "rb")
            self._fh[path] = f
        f.seek(offs[i])
        return f.readline().decode("utf-8").rstrip("\n")

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return (self._read(self.src_file, self.src_off, i),
                self._read(self.tgt_file, self.tgt_off, i))


class BucketBatchSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.ids = self.data_source.get_valid()




    def __iter__(self):
        n = len(self.ids)
        full = n - n % 16                       # 整块能覆盖到的位置
        starts = list(range(0, full, 16))
        random.shuffle(starts)
        max_index  = starts.index(max(starts)-16)
        starts[max_index] , starts[0] = starts[0] , starts[max_index]
        v_idx = []
        for i in starts:                        # 整块：每块 16 条，长度相近
            v_idx.extend(self.ids[i:i + 16])
        v_idx.extend(self.ids[full:])           # 余数放最后，不影响前面切批对齐
        return iter(v_idx)

    def __len__(self):
        # 这个 Sampler 交给 DataLoader(sampler=..., batch_size=16) 用，
        # DataLoader 会自己用 ceil(len(sampler)/batch_size) 算批数，
        # 所以这里必须返回「样本数」，不是批数。
        return len(self.ids)




def collate_fn(batch):
    src_language_list = [item[0] for item in batch]
    target_language_list = [item[1] for item in batch]
    # ----------------
    # 2.分词
    # ----------------
    src_encoding = [tokenizer.encode(item) for item in src_language_list]
    target_encoding = [tokenizer.encode("<bos> "+item) for item in target_language_list]
    label_encoding = [tokenizer.encode(item+" <eos>") for item in target_language_list]
    # ----------------
    # 3.获取tokens
    # ----------------
    #src_tokens = [item.tokens for item in src_encoding]
    # target_tokens = [item.tokens for item in target_encoding]
    # labels_tokens = [item.tokens for item in label_encoding]


    # ----------------
    # 4.获取ids
    # ----------------
    src_ids = [item.ids for item in src_encoding]
    target_ids = [item.ids for item in target_encoding]
    labels_ids = [item.ids for item in label_encoding]

    # ---------------------------
    # 5.获取src和target的最长序列长度
    # ---------------------------
    max_srclen = max([len(item) for item in src_ids])
    max_tgtnlen = max([len(item) for item in target_ids])
    max_labelslen = max([len(item) for item in labels_ids])

    # --------------------------------
    # 6.对所有序列进行<PAD>填充保持一致长度
    # --------------------------------
    #6.1 填充tokens
    # src_tokens = [item + [" <pad>" for i in range((max_srclen - len(item)))] for item in src_tokens]
    # target_tokens = [item + [" <pad>" for i in range((max_tgtnlen - len(item)))] for item in target_tokens]
    #6.2 填充ids
    PAD = tokenizer.token_to_id("<pad>")
    src_ids = [item + [PAD for i in range((max_srclen - len(item)))] for item in src_ids]
    target_ids = [item + [PAD for i in range((max_tgtnlen - len(item)))] for item in target_ids]
    labels_ids = [item + [PAD for i in range((max_labelslen - len(item)))] for item in labels_ids]


    # ---------------------------
    # 7.转换为tensor
    # ---------------------------
    src_ids = torch.tensor(src_ids)
    target_ids = torch.tensor(target_ids)
    labels_ids = torch.tensor(labels_ids)

    # ---------------------------
    # 8.获取Mask
    # ---------------------------
    src_mask = torch.zeros((src_ids.size(0), src_ids.size(1)))
    src_mask[(src_ids==tokenizer.token_to_id("<pad>"))] = float("-inf")
    # target_mask = torch.zeros((target_ids.size(0), target_ids.size(1)))
    # target_mask[(target_ids==tokenizer.token_to_id("<pad>"))] = float("-inf")
    labels_mask = torch.ones((labels_ids.size(0), labels_ids.size(1)))
    labels_mask[(labels_ids==tokenizer.token_to_id("<pad>"))] = 0



    return src_ids, target_ids, labels_ids , src_mask, labels_mask








# if __name__ == "__main__":
    # with open("data/wmt14_en_de/train.en", "r", encoding="utf-8") as f:
    #    src_language_list = ["<bos> " + line.strip("\n")  + " <eos>"for line in f.readlines()[0:10]]
    # print(src_language_list)
    # dataset = MyDataSet("data/wmt14_en_de/train.en", "data/wmt14_en_de/train.de")
    # dataloder = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    # for i, batch in enumerate(dataloder):
    #     print(i)
    #     # src_embedding, target_embedding,labels = batch
    #     #print(src_embedding.shape, target_embedding.shape, labels.shape)
    #     if i >10:
    #         break



