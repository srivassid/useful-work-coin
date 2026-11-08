"""A generation model's weights tree: detrt.stage.weight_rows, then detgen.head.head_weight_rows
(the final norm and lm_head), committed like detrt.stage.weights_tree."""
from __future__ import annotations

import itertools
import struct

from detrt.qwen3 import DetQwen3
from detrt.stage import _leaf_hash, merkle_root, row_root, weight_rows

from .head import head_weight_rows


def gen_weight_rows(m: DetQwen3):
    return itertools.chain(weight_rows(m), head_weight_rows(m))


def gen_weights_tree(m: DetQwen3) -> tuple[bytes, list[bytes]]:
    hs = [_leaf_hash(b"w" + struct.pack("<I", i) + row_root(row)) for i, (_, row) in enumerate(gen_weight_rows(m))]
    return merkle_root(hs), hs
