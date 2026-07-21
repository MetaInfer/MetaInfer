#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Minimal Qwen3/Qwen2 byte-level BPE tokenizer contract.
//
// The GGUF loader owns file parsing and fills this structure from:
//   tokenizer.ggml.model
//   tokenizer.ggml.pre
//   tokenizer.ggml.tokens
//   tokenizer.ggml.merges
//   tokenizer.ggml.token_type
//   tokenizer.ggml.{bos,eos,padding}_token_id
//
// Qwen3 input is expected to be valid UTF-8 in NFC form.  The implementation
// intentionally has no ICU/utf8proc dependency, so it cannot normalize a
// decomposed Unicode string to NFC by itself.
struct Qwen3TokenizerData {
    std::string model;
    std::string pre_tokenizer;
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> token_types;

    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    int32_t pad_token_id = -1;
    bool add_bos_token = false;
};

struct Qwen3EncodeOptions {
    // Recognize CONTROL and USER_DEFINED GGUF tokens such as <|im_start|>
    // atomically.  Chat prompts must enable this.
    bool parse_special = true;
    bool add_bos = false;
    bool add_eos = false;
};

class Qwen3Tokenizer {
public:
    // GGML vocabulary token types stored in tokenizer.ggml.token_type.
    enum TokenType : int32_t {
        TOKEN_TYPE_NORMAL       = 1,
        TOKEN_TYPE_UNKNOWN      = 2,
        TOKEN_TYPE_CONTROL      = 3,
        TOKEN_TYPE_USER_DEFINED = 4,
        TOKEN_TYPE_UNUSED       = 5,
        TOKEN_TYPE_BYTE         = 6,
    };

    bool load(const Qwen3TokenizerData& data, std::string* error = nullptr);

    bool ready() const { return ready_; }
    size_t vocab_size() const { return id_to_token_.size(); }

    std::vector<int32_t> encode(
            const std::string& text,
            const Qwen3EncodeOptions& options = {}) const;

    // Compatibility with the previous notebook API.  Qwen3 normally has no
    // BOS, so add_bos only has an effect when GGUF provides a valid BOS id.
    std::vector<int32_t> encode(
            const std::string& text,
            bool add_bos,
            bool add_eos) const;

    std::string decode(
            const std::vector<int32_t>& token_ids,
            bool skip_special = true) const;

    std::string id_word(int32_t token_id) const;
    int32_t token_id(const std::string& token) const;

    int32_t bos_token_id() const { return bos_token_id_; }
    int32_t eos_token_id() const { return eos_token_id_; }
    int32_t pad_token_id() const { return pad_token_id_; }

    // Minimal single-turn form of the official Qwen3 chat template.  The
    // returned string must be encoded with parse_special=true.
    std::string format_chat_prompt(
            const std::string& user_text,
            const std::string& system_text = {},
            bool enable_thinking = true) const;

private:
    struct MergePair {
        std::string first;
        std::string second;

        bool operator==(const MergePair& other) const {
            return first == other.first && second == other.second;
        }
    };

    struct MergePairHash {
        size_t operator()(const MergePair& pair) const;
    };

    struct Utf8Unit {
        uint32_t codepoint;
        size_t begin;
        size_t end;
    };

    void clear();
    void build_byte_maps();

    std::vector<int32_t> encode_normal(const std::string& text) const;
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& piece) const;

    static std::vector<Utf8Unit> utf8_units(const std::string& text);
    static std::string utf8_from_codepoint(uint32_t codepoint);
    static bool is_letter(uint32_t codepoint);
    static bool is_number(uint32_t codepoint);
    static bool is_space(uint32_t codepoint);
    static bool is_newline(uint32_t codepoint);
    static bool is_punctuation_or_symbol(uint32_t codepoint);

    bool is_atomic_token(int32_t token_id) const;
    bool is_control_token(int32_t token_id) const;

private:
    bool ready_ = false;
    bool add_bos_token_ = false;

    int32_t bos_token_id_ = -1;
    int32_t eos_token_id_ = -1;
    int32_t pad_token_id_ = -1;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<MergePair, int32_t, MergePairHash> merge_ranks_;

    std::unordered_set<int32_t> atomic_token_ids_;
    std::unordered_set<int32_t> control_token_ids_;
    std::vector<std::pair<std::string, int32_t>> atomic_tokens_by_length_;

    std::array<std::string, 256> byte_encoder_;
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
};

// Keep the old class spelling usable by small examples that included the
// original tokenizer.hpp.
using tokenizer = Qwen3Tokenizer;
