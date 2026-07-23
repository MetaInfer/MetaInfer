"""Compile and exercise the standalone Qwen3 byte-level BPE tokenizer."""

from __future__ import annotations

import subprocess
from pathlib import Path


TASK_DIR = Path(__file__).parents[1]
TOKENIZER_CPP = TASK_DIR / "notebooks" / "reference" / "tokenizer.cpp"
TOKENIZER_HPP = TASK_DIR / "notebooks" / "reference" / "tokenizer.hpp"
LOADER_NOTES = TASK_DIR / "notebooks" / "formats" / "gguf" / "qwen3_loader.md"


def test_qwen3_tokenizer_source_uses_gguf_bpe_contract():
    header = TOKENIZER_HPP.read_text(encoding="utf-8")
    source = TOKENIZER_CPP.read_text(encoding="utf-8")
    loader_notes = LOADER_NOTES.read_text(encoding="utf-8")

    assert "Qwen3TokenizerData" in header
    assert "tokenizer.ggml.tokens" in header
    assert "tokenizer.ggml.merges" in header
    assert "merge_ranks_" in header
    assert "byte_encoder_" in header
    assert "format_chat_prompt" in header

    assert 'data.model != "gpt2"' in source
    assert "add_prefix_space" not in source
    assert "vocab_scores" not in source
    assert "tokens.push_back(1)" not in source
    assert "tokens.push_back(2)" not in source
    assert "Qwen3TokenizerData" in loader_notes
    assert "不要再生成或读取旧式 `tokenizer.bin`" in loader_notes


def test_qwen3_tokenizer_compiles_and_round_trips_utf8(tmp_path: Path):
    harness = tmp_path / "tokenizer_test.cpp"
    binary = tmp_path / "tokenizer_test"
    harness.write_text(
        r'''
#include "tokenizer.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

static std::string utf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}

static std::array<std::string, 256> byte_tokens() {
    std::vector<int> bytes;
    std::vector<uint32_t> codepoints;
    for (int value = '!'; value <= '~'; ++value) {
        bytes.push_back(value);
        codepoints.push_back(value);
    }
    for (int value = 0xa1; value <= 0xac; ++value) {
        bytes.push_back(value);
        codepoints.push_back(value);
    }
    for (int value = 0xae; value <= 0xff; ++value) {
        bytes.push_back(value);
        codepoints.push_back(value);
    }
    std::array<bool, 256> present{};
    for (int value : bytes) present[value] = true;
    uint32_t extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (!present[value]) {
            bytes.push_back(value);
            codepoints.push_back(256 + extra++);
        }
    }
    std::array<std::string, 256> result;
    for (size_t i = 0; i < bytes.size(); ++i) {
        result[bytes[i]] = utf8(codepoints[i]);
    }
    return result;
}

int main() {
    Qwen3TokenizerData data;
    data.model = "gpt2";
    const auto base = byte_tokens();
    data.tokens.assign(base.begin(), base.end());
    data.token_types.assign(256, Qwen3Tokenizer::TOKEN_TYPE_NORMAL);

    data.tokens.push_back("he");       // 256
    data.tokens.push_back("ll");       // 257
    data.tokens.push_back("hell");     // 258
    data.tokens.push_back("hello");    // 259
    data.tokens.push_back(base[' '] + std::string("hello")); // 260
    data.tokens.push_back("<|im_start|>"); // 261
    data.tokens.push_back("<|im_end|>");   // 262
    data.token_types.resize(263, Qwen3Tokenizer::TOKEN_TYPE_NORMAL);
    data.token_types[261] = Qwen3Tokenizer::TOKEN_TYPE_CONTROL;
    data.token_types[262] = Qwen3Tokenizer::TOKEN_TYPE_CONTROL;
    data.eos_token_id = 262;

    data.merges = {
        "h e",
        "l l",
        "he ll",
        "hell o",
        base[' '] + std::string(" hello"),
    };

    Qwen3Tokenizer tokenizer;
    std::string error;
    assert(tokenizer.load(data, &error));
    assert(tokenizer.ready());
    assert(tokenizer.bos_token_id() == -1);
    assert(tokenizer.eos_token_id() == 262);

    const auto hello = tokenizer.encode("hello");
    assert((hello == std::vector<int32_t>{259}));
    const auto spaced = tokenizer.encode(" hello");
    assert((spaced == std::vector<int32_t>{260}));

    const std::string chinese = "你好";
    const auto chinese_ids = tokenizer.encode(chinese);
    assert(!chinese_ids.empty());
    assert(tokenizer.decode(chinese_ids) == chinese);

    const auto special = tokenizer.encode(
        "<|im_start|>hello<|im_end|>");
    assert((special == std::vector<int32_t>{261, 259, 262}));
    assert(tokenizer.decode(special, true) == "hello");
    assert(tokenizer.decode(special, false)
        == "<|im_start|>hello<|im_end|>");

    const auto with_eos = tokenizer.encode("hello", false, true);
    assert((with_eos == std::vector<int32_t>{259, 262}));

    const std::string prompt = tokenizer.format_chat_prompt(
        "hi", "", false);
    assert(prompt.find("<|im_start|>user\nhi<|im_end|>\n") == 0);
    assert(prompt.find("<think>\n\n</think>\n\n") != std::string::npos);
    return 0;
}
''',
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(TOKENIZER_HPP.parent),
            str(harness),
            str(TOKENIZER_CPP),
            "-o",
            str(binary),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(binary)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr
