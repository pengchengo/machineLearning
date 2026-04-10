# encoding: utf-8
"""
Multi30k 英德平行语料加载，供 Transformer 训练使用。
数据目录：本文件同级的 multi30k/（train.en / train.de, val.en / val.de）

用法示例::

    from learnTransformer.scripts.DataLoader import get_dataloaders

    train_loader, val_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=32)
    for src, tgt in train_loader:
        # src: [batch, src_len], tgt: [batch, tgt_len]，含 <sos>...<eos>
        dec_in = tgt[:, :-1]
        dec_out = tgt[:, 1:]
        ...
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from learnTransformer.scripts.Config import (
    EOS,
    EOS_IDX,
    PAD,
    PAD_IDX,
    SOS,
    SOS_IDX,
    UNK,
    UNK_IDX,
)

# 数据目录：scripts/multi30k
_MULTI30K_DIR = Path(__file__).resolve().parent / "multi30k"


class Vocabulary:
    """词表：英文/德文各建一份。"""

    def __init__(self) -> None:
        specials = [PAD, UNK, SOS, EOS]
        self.word2idx: dict[str, int] = {w: i for i, w in enumerate(specials)}
        self.idx2word: List[str] = list(specials)

    def __len__(self) -> int:
        return len(self.idx2word)

    def _add_word(self, w: str) -> None:
        if w not in self.word2idx:
            i = len(self.idx2word)
            self.word2idx[w] = i
            self.idx2word.append(w)

    @classmethod
    def build(cls, sentences: List[str], min_freq: int = 2) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for line in sentences:
            counter.update(line.strip().split())
        v = cls()
        for w, c in counter.most_common():
            if c >= min_freq:
                v._add_word(w)
        return v

    def encode(self, text: str, *, add_sos_eos: bool = False) -> List[int]:
        toks = text.strip().split()
        ids = [self.word2idx.get(t, UNK_IDX) for t in toks]
        if add_sos_eos:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids


class Multi30kDataset(Dataset):
    """一行英文对应一行德文。"""

    def __init__(
        self,
        src_path: Path,
        tgt_path: Path,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: Optional[int] = None,
    ) -> None:
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pairs: List[Tuple[str, str]] = []
        with open(src_path, encoding="utf-8") as fs, open(tgt_path, encoding="utf-8") as ft:
            for ls, lt in zip(fs, ft):
                ls, lt = ls.strip(), lt.strip()
                if not ls or not lt:
                    continue
                if max_len is not None:
                    if len(ls.split()) > max_len or len(lt.split()) > max_len:
                        continue
                self.pairs.append((ls, lt))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ls, lt = self.pairs[idx]
        src_ids = self.src_vocab.encode(ls, add_sos_eos=False)
        # 目标：<sos> ... <eos>，便于切片得到 decoder 输入/标签
        tgt_ids = self.tgt_vocab.encode(lt, add_sos_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def _collate_pad(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_list, tgt_list = zip(*batch)
    max_src = max(s.size(0) for s in src_list)
    max_tgt = max(t.size(0) for t in tgt_list)
    bsz = len(batch)
    src_pad = torch.full((bsz, max_src), PAD_IDX, dtype=torch.long)
    tgt_pad = torch.full((bsz, max_tgt), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_pad[i, : s.size(0)] = s
        tgt_pad[i, : t.size(0)] = t
    return src_pad, tgt_pad


def _read_lines(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_dataloaders(
    batch_size: int = 32,
    min_freq: int = 2,
    max_len: Optional[int] = None,
    num_workers: int = 0,
    src_lang: str = "en",
    tgt_lang: str = "de",
    data_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    构建训练集 / 验证集 DataLoader 及词表。

    :param batch_size: 批大小
    :param min_freq: 词频阈值，低于此频次的词记为 <unk>
    :param max_len: 若设置，过滤掉源或目标词数超过该值的句子
    :param num_workers: DataLoader worker 数（Windows 下建议 0）
    :param src_lang: 源语言文件后缀，如 en
    :param tgt_lang: 目标语言文件后缀，如 de
    :param data_dir: multi30k 目录，默认 scripts/multi30k
    :return: train_loader, val_loader, src_vocab, tgt_vocab
    """
    root = data_dir if data_dir is not None else _MULTI30K_DIR
    train_src = root / f"train.{src_lang}"
    train_tgt = root / f"train.{tgt_lang}"
    val_src = root / f"val.{src_lang}"
    val_tgt = root / f"val.{tgt_lang}"

    for p in (train_src, train_tgt, val_src, val_tgt):
        if not p.exists():
            raise FileNotFoundError(f"找不到数据文件: {p}")

    train_src_lines = _read_lines(train_src)
    train_tgt_lines = _read_lines(train_tgt)
    if len(train_src_lines) != len(train_tgt_lines):
        raise ValueError(
            f"train 行数不一致: {len(train_src_lines)} vs {len(train_tgt_lines)}"
        )

    src_vocab = Vocabulary.build(train_src_lines, min_freq=min_freq)
    tgt_vocab = Vocabulary.build(train_tgt_lines, min_freq=min_freq)

    train_ds = Multi30kDataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len=max_len)
    val_ds = Multi30kDataset(val_src, val_tgt, src_vocab, tgt_vocab, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_pad,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_pad,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, src_vocab, tgt_vocab


def get_test_loader(
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    batch_size: int = 32,
    num_workers: int = 0,
    src_lang: str = "en",
    tgt_lang: str = "de",
    data_dir: Optional[Path] = None,
    split: str = "test2016",
) -> DataLoader:
    """测试集（如 test2016），需与训练时同一套词表。"""
    root = data_dir if data_dir is not None else _MULTI30K_DIR
    src_p = root / f"{split}.{src_lang}"
    tgt_p = root / f"{split}.{tgt_lang}"
    if not src_p.exists() or not tgt_p.exists():
        raise FileNotFoundError(f"测试文件不存在: {src_p} / {tgt_p}")
    ds = Multi30kDataset(src_p, tgt_p, src_vocab, tgt_vocab, max_len=None)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_pad,
        pin_memory=torch.cuda.is_available(),
    )
