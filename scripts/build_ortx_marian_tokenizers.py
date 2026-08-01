#!/usr/bin/env python3
"""Generate deterministic Marian tokenizer graphs for ONNX Runtime Extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import sentencepiece as spm
from onnx import TensorProto, helper, numpy_helper


def build(runtime_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    vocab = json.loads((runtime_dir / "vocab.json").read_text(encoding="utf-8"))
    source_model = runtime_dir / "source.spm"
    target_model = runtime_dir / "target.spm"
    source = spm.SentencePieceProcessor(model_file=str(source_model))
    target = spm.SentencePieceProcessor(model_file=str(target_model))

    source_to_marian = np.array(
        [vocab.get(source.id_to_piece(index), vocab["<unk>"]) for index in range(source.get_piece_size())],
        dtype=np.int64,
    )
    target_piece_to_id = {
        target.id_to_piece(index): index for index in range(target.get_piece_size())
    }
    marian_to_target = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
    for piece, marian_id in vocab.items():
        marian_to_target[marian_id] = target_piece_to_id.get(piece, 0)

    source_initializers = [
        numpy_helper.from_array(np.array([0], np.int64), "nbest_size"),
        numpy_helper.from_array(np.array([0.0], np.float32), "alpha"),
        numpy_helper.from_array(np.array([False], np.bool_), "add_bos"),
        numpy_helper.from_array(np.array([False], np.bool_), "add_eos"),
        numpy_helper.from_array(np.array([False], np.bool_), "reverse"),
        numpy_helper.from_array(np.array([False], np.bool_), "fairseq"),
        numpy_helper.from_array(source_to_marian, "src_to_marian"),
        numpy_helper.from_array(np.array([0], np.int64), "eos"),
        numpy_helper.from_array(np.array([0], np.int64), "axes0"),
    ]
    source_nodes = [
        helper.make_node(
            "SentencepieceTokenizer",
            ["text", "nbest_size", "alpha", "add_bos", "add_eos", "reverse", "fairseq"],
            ["spm_ids_i32", "instance_indices", "token_indices"],
            domain="com.microsoft.extensions",
            model=source_model.read_bytes(),
        ),
        helper.make_node("Cast", ["spm_ids_i32"], ["spm_ids"], to=TensorProto.INT64),
        helper.make_node("Gather", ["src_to_marian", "spm_ids"], ["mapped"], axis=0),
        helper.make_node("Concat", ["mapped", "eos"], ["flat_ids"], axis=0),
        helper.make_node("Unsqueeze", ["flat_ids", "axes0"], ["input_ids"]),
        helper.make_node("Shape", ["input_ids"], ["id_shape"]),
        helper.make_node(
            "ConstantOfShape",
            ["id_shape"],
            ["attention_mask"],
            value=numpy_helper.from_array(np.array([1], np.int64)),
        ),
    ]
    source_graph = helper.make_graph(
        source_nodes,
        "MarianSourceTokenizer",
        [helper.make_tensor_value_info("text", TensorProto.STRING, [1])],
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, None]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, None]),
        ],
        initializer=source_initializers,
    )
    source_onnx = helper.make_model(
        source_graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft.extensions", 1)],
        producer_name="manga-tokenizer-feasibility-proof",
        ir_version=9,
    )

    target_initializers = [
        numpy_helper.from_array(np.array([0], np.int64), "eos_id"),
        numpy_helper.from_array(np.array([1], np.int64), "unk_id"),
        numpy_helper.from_array(np.array([61917], np.int64), "pad_id"),
        numpy_helper.from_array(np.array([False], np.bool_), "fairseq"),
        numpy_helper.from_array(marian_to_target, "marian_to_tgt"),
    ]
    target_nodes = [
        helper.make_node("Equal", ["generated_ids", "eos_id"], ["is_eos"]),
        helper.make_node("Equal", ["generated_ids", "unk_id"], ["is_unk"]),
        helper.make_node("Equal", ["generated_ids", "pad_id"], ["is_pad"]),
        helper.make_node("Or", ["is_eos", "is_unk"], ["special01"]),
        helper.make_node("Or", ["special01", "is_pad"], ["special"]),
        helper.make_node("Not", ["special"], ["keep"]),
        helper.make_node("Compress", ["generated_ids", "keep"], ["content_ids"], axis=0),
        helper.make_node("Gather", ["marian_to_tgt", "content_ids"], ["target_spm_ids"], axis=0),
        helper.make_node(
            "SentencepieceDecoder",
            ["target_spm_ids", "fairseq"],
            ["text"],
            domain="com.microsoft.extensions",
            model=target_model.read_bytes(),
        ),
    ]
    target_graph = helper.make_graph(
        target_nodes,
        "MarianTargetDetokenizer",
        [helper.make_tensor_value_info("generated_ids", TensorProto.INT64, [None])],
        [helper.make_tensor_value_info("text", TensorProto.STRING, [1])],
        initializer=target_initializers,
    )
    target_onnx = helper.make_model(
        target_graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft.extensions", 1)],
        producer_name="manga-tokenizer-feasibility-proof",
        ir_version=9,
    )

    onnx.checker.check_model(source_onnx)
    onnx.checker.check_model(target_onnx)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_output = output_dir / "marian_source_tokenizer.onnx"
    target_output = output_dir / "marian_target_detokenizer.onnx"
    onnx.save(source_onnx, source_output)
    onnx.save(target_onnx, target_output)
    return source_output, target_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in build(args.runtime_dir.resolve(), args.output_dir.resolve()):
        print(f"generated={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
