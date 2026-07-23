#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace {

bool starts_with_at(
        const std::string& text,
        size_t offset,
        const std::string& needle) {
    return offset <= text.size()
            && needle.size() <= text.size() - offset
            && text.compare(offset, needle.size(), needle) == 0;
}

char ascii_lower(char value) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
}

bool ascii_case_equal_at(
        const std::string& text,
        size_t offset,
        const char* needle) {
    size_t length = 0;
    while (needle[length] != '\0') {
        ++length;
    }
    if (offset > text.size() || length > text.size() - offset) {
        return false;
    }
    for (size_t i = 0; i < length; ++i) {
        if (ascii_lower(text[offset + i]) != ascii_lower(needle[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

size_t Qwen3Tokenizer::MergePairHash::operator()(
        const MergePair& pair) const {
    const size_t h1 = std::hash<std::string>{}(pair.first);
    const size_t h2 = std::hash<std::string>{}(pair.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
}

void Qwen3Tokenizer::clear() {
    ready_ = false;
    add_bos_token_ = false;
    bos_token_id_ = -1;
    eos_token_id_ = -1;
    pad_token_id_ = -1;

    id_to_token_.clear();
    token_to_id_.clear();
    merge_ranks_.clear();
    atomic_token_ids_.clear();
    control_token_ids_.clear();
    atomic_tokens_by_length_.clear();
    byte_decoder_.clear();
    for (std::string& token : byte_encoder_) {
        token.clear();
    }
}

std::string Qwen3Tokenizer::utf8_from_codepoint(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

void Qwen3Tokenizer::build_byte_maps() {
    std::vector<int> bytes;
    std::vector<uint32_t> codepoints;

    for (int value = '!'; value <= '~'; ++value) {
        bytes.push_back(value);
        codepoints.push_back(static_cast<uint32_t>(value));
    }
    for (int value = 0xA1; value <= 0xAC; ++value) {
        bytes.push_back(value);
        codepoints.push_back(static_cast<uint32_t>(value));
    }
    for (int value = 0xAE; value <= 0xFF; ++value) {
        bytes.push_back(value);
        codepoints.push_back(static_cast<uint32_t>(value));
    }

    std::array<bool, 256> present{};
    for (int value : bytes) {
        present[static_cast<size_t>(value)] = true;
    }

    uint32_t extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (!present[static_cast<size_t>(value)]) {
            bytes.push_back(value);
            codepoints.push_back(256 + extra);
            ++extra;
        }
    }

    for (size_t i = 0; i < bytes.size(); ++i) {
        const uint8_t byte = static_cast<uint8_t>(bytes[i]);
        const uint32_t cp = codepoints[i];
        byte_encoder_[byte] = utf8_from_codepoint(cp);
        byte_decoder_[cp] = byte;
    }
}

bool Qwen3Tokenizer::load(
        const Qwen3TokenizerData& data,
        std::string* error) {
    clear();

    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        clear();
        return false;
    };

    if (!data.model.empty() && data.model != "gpt2") {
        return fail("Qwen3 tokenizer requires tokenizer.ggml.model=gpt2, got: "
                + data.model);
    }
    if (data.tokens.empty()) {
        return fail("tokenizer.ggml.tokens is empty");
    }
    if (!data.token_types.empty()
            && data.token_types.size() != data.tokens.size()) {
        return fail("tokenizer.ggml.token_type length does not match tokens");
    }

    build_byte_maps();
    id_to_token_ = data.tokens;
    token_to_id_.reserve(id_to_token_.size());

    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        const std::string& token = id_to_token_[i];
        if (token_to_id_.find(token) != token_to_id_.end()) {
            return fail("duplicate tokenizer token at id " + std::to_string(i));
        }
        token_to_id_[token] = static_cast<int32_t>(i);

        const int32_t type = data.token_types.empty()
                ? TOKEN_TYPE_NORMAL
                : data.token_types[i];
        if (type == TOKEN_TYPE_CONTROL || type == TOKEN_TYPE_USER_DEFINED) {
            atomic_token_ids_.insert(static_cast<int32_t>(i));
        }
        if (type == TOKEN_TYPE_CONTROL) {
            control_token_ids_.insert(static_cast<int32_t>(i));
        }
    }

    merge_ranks_.reserve(data.merges.size());
    for (size_t rank = 0; rank < data.merges.size(); ++rank) {
        const std::string& merge = data.merges[rank];
        const size_t separator = merge.find(' ');
        if (separator == std::string::npos || separator == 0
                || separator + 1 >= merge.size()) {
            return fail("invalid BPE merge at rank " + std::to_string(rank));
        }

        MergePair pair{
                merge.substr(0, separator),
                merge.substr(separator + 1)};
        merge_ranks_.emplace(std::move(pair), static_cast<int32_t>(rank));
    }

    auto validate_id = [&](int32_t id, const char* name) -> bool {
        if (id < -1 || id >= static_cast<int32_t>(id_to_token_.size())) {
            if (error != nullptr) {
                *error = std::string(name) + " is outside tokenizer vocabulary";
            }
            return false;
        }
        return true;
    };

    if (!validate_id(data.bos_token_id, "bos_token_id")
            || !validate_id(data.eos_token_id, "eos_token_id")
            || !validate_id(data.pad_token_id, "pad_token_id")) {
        clear();
        return false;
    }

    bos_token_id_ = data.bos_token_id;
    eos_token_id_ = data.eos_token_id;
    pad_token_id_ = data.pad_token_id;
    add_bos_token_ = data.add_bos_token;

    // BOS/EOS/PAD are special by definition.  Keep them atomic and skippable
    // even when an incomplete converter omitted tokenizer.ggml.token_type.
    for (int32_t id : {bos_token_id_, eos_token_id_, pad_token_id_}) {
        if (id >= 0) {
            atomic_token_ids_.insert(id);
            control_token_ids_.insert(id);
        }
    }

    atomic_tokens_by_length_.reserve(atomic_token_ids_.size());
    for (int32_t id : atomic_token_ids_) {
        atomic_tokens_by_length_.emplace_back(id_to_token_[id], id);
    }
    std::sort(
            atomic_tokens_by_length_.begin(),
            atomic_tokens_by_length_.end(),
            [](const auto& lhs, const auto& rhs) {
                if (lhs.first.size() != rhs.first.size()) {
                    return lhs.first.size() > rhs.first.size();
                }
                return lhs.first < rhs.first;
            });

    ready_ = true;
    return true;
}

int32_t Qwen3Tokenizer::token_id(const std::string& token) const {
    const auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

std::string Qwen3Tokenizer::id_word(int32_t token_id_value) const {
    if (token_id_value < 0
            || token_id_value >= static_cast<int32_t>(id_to_token_.size())) {
        return "<UNK>";
    }
    return id_to_token_[token_id_value];
}

bool Qwen3Tokenizer::is_atomic_token(int32_t token_id_value) const {
    return atomic_token_ids_.find(token_id_value) != atomic_token_ids_.end();
}

bool Qwen3Tokenizer::is_control_token(int32_t token_id_value) const {
    return control_token_ids_.find(token_id_value) != control_token_ids_.end();
}

std::vector<Qwen3Tokenizer::Utf8Unit> Qwen3Tokenizer::utf8_units(
        const std::string& text) {
    std::vector<Utf8Unit> units;
    size_t i = 0;
    while (i < text.size()) {
        const size_t begin = i;
        const uint8_t first = static_cast<uint8_t>(text[i]);
        uint32_t cp = first;
        size_t length = 1;

        if ((first & 0xE0) == 0xC0 && i + 1 < text.size()) {
            cp = first & 0x1F;
            length = 2;
        } else if ((first & 0xF0) == 0xE0 && i + 2 < text.size()) {
            cp = first & 0x0F;
            length = 3;
        } else if ((first & 0xF8) == 0xF0 && i + 3 < text.size()) {
            cp = first & 0x07;
            length = 4;
        }

        bool valid = length > 1;
        for (size_t j = 1; j < length && valid; ++j) {
            const uint8_t next = static_cast<uint8_t>(text[i + j]);
            if ((next & 0xC0) != 0x80) {
                valid = false;
                break;
            }
            cp = (cp << 6) | (next & 0x3F);
        }
        if (!valid && length > 1) {
            cp = first;
            length = 1;
        }

        i += length;
        units.push_back({cp, begin, i});
    }
    return units;
}

bool Qwen3Tokenizer::is_newline(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

bool Qwen3Tokenizer::is_space(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r'
            || cp == '\v' || cp == '\f') {
        return true;
    }
    return cp == 0x0085 || cp == 0x00A0 || cp == 0x1680
            || (cp >= 0x2000 && cp <= 0x200A)
            || cp == 0x2028 || cp == 0x2029 || cp == 0x202F
            || cp == 0x205F || cp == 0x3000;
}

bool Qwen3Tokenizer::is_number(uint32_t cp) {
    if (cp >= '0' && cp <= '9') {
        return true;
    }
    return (cp >= 0x0660 && cp <= 0x0669)
            || (cp >= 0x06F0 && cp <= 0x06F9)
            || (cp >= 0x0966 && cp <= 0x096F)
            || (cp >= 0x09E6 && cp <= 0x09EF)
            || (cp >= 0x0E50 && cp <= 0x0E59)
            || (cp >= 0xFF10 && cp <= 0xFF19);
}

bool Qwen3Tokenizer::is_punctuation_or_symbol(uint32_t cp) {
    if (cp < 128) {
        return std::ispunct(static_cast<unsigned char>(cp)) != 0;
    }
    return (cp >= 0x2000 && cp <= 0x206F)
            || (cp >= 0x20A0 && cp <= 0x20CF)
            || (cp >= 0x2100 && cp <= 0x214F)
            || (cp >= 0x2190 && cp <= 0x2BFF)
            || (cp >= 0x3000 && cp <= 0x303F)
            || (cp >= 0xFE10 && cp <= 0xFE1F)
            || (cp >= 0xFE30 && cp <= 0xFE6F)
            || (cp >= 0xFF01 && cp <= 0xFF0F)
            || (cp >= 0xFF1A && cp <= 0xFF20)
            || (cp >= 0xFF3B && cp <= 0xFF40)
            || (cp >= 0xFF5B && cp <= 0xFF65)
            || (cp >= 0x1F000 && cp <= 0x1FAFF);
}

bool Qwen3Tokenizer::is_letter(uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) {
        return true;
    }
    if (cp < 128 || is_space(cp) || is_number(cp)
            || is_punctuation_or_symbol(cp)) {
        return false;
    }
    // C++17 has no Unicode general-category API.  Treat remaining valid
    // non-ASCII codepoints as letters/marks, which covers common Qwen3 input
    // scripts (CJK, kana, hangul, Cyrillic, Arabic and Devanagari).
    return cp >= 0x80 && !(cp >= 0x7F && cp <= 0x9F);
}

std::vector<std::string> Qwen3Tokenizer::pretokenize(
        const std::string& text) const {
    const std::vector<Utf8Unit> units = utf8_units(text);
    std::vector<std::string> pieces;
    size_t i = 0;

    auto add_piece = [&](size_t begin_unit, size_t end_unit) {
        if (begin_unit >= end_unit) {
            return;
        }
        pieces.push_back(text.substr(
                units[begin_unit].begin,
                units[end_unit - 1].end - units[begin_unit].begin));
    };

    while (i < units.size()) {
        const size_t byte_offset = units[i].begin;

        // Qwen regex first alternative: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        const char* contraction = nullptr;
        for (const char* candidate : {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"}) {
            if (ascii_case_equal_at(text, byte_offset, candidate)) {
                contraction = candidate;
                break;
            }
        }
        if (contraction != nullptr) {
            size_t length = 0;
            while (contraction[length] != '\0') {
                ++length;
            }
            const size_t end_byte = byte_offset + length;
            size_t j = i;
            while (j < units.size() && units[j].end <= end_byte) {
                ++j;
            }
            add_piece(i, j);
            i = j;
            continue;
        }

        if (is_letter(units[i].codepoint)) {
            size_t j = i + 1;
            while (j < units.size() && is_letter(units[j].codepoint)) {
                ++j;
            }
            add_piece(i, j);
            i = j;
            continue;
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!is_newline(units[i].codepoint)
                && !is_letter(units[i].codepoint)
                && !is_number(units[i].codepoint)
                && i + 1 < units.size()
                && is_letter(units[i + 1].codepoint)) {
            size_t j = i + 2;
            while (j < units.size() && is_letter(units[j].codepoint)) {
                ++j;
            }
            add_piece(i, j);
            i = j;
            continue;
        }

        // \p{N}: Qwen intentionally isolates one Unicode number at a time.
        if (is_number(units[i].codepoint)) {
            add_piece(i, i + 1);
            ++i;
            continue;
        }

        // Optional ASCII space followed by a run of non-space symbols.
        size_t symbol_begin = i;
        size_t symbol = i;
        if (units[symbol].codepoint == ' ' && symbol + 1 < units.size()
                && !is_space(units[symbol + 1].codepoint)
                && !is_letter(units[symbol + 1].codepoint)
                && !is_number(units[symbol + 1].codepoint)) {
            ++symbol;
        }
        if (symbol < units.size()
                && !is_space(units[symbol].codepoint)
                && !is_letter(units[symbol].codepoint)
                && !is_number(units[symbol].codepoint)) {
            size_t j = symbol + 1;
            while (j < units.size()
                    && !is_space(units[j].codepoint)
                    && !is_letter(units[j].codepoint)
                    && !is_number(units[j].codepoint)) {
                ++j;
            }
            while (j < units.size() && is_newline(units[j].codepoint)) {
                ++j;
            }
            add_piece(symbol_begin, j);
            i = j;
            continue;
        }

        if (is_space(units[i].codepoint)) {
            size_t run_end = i + 1;
            size_t last_newline = std::numeric_limits<size_t>::max();
            if (is_newline(units[i].codepoint)) {
                last_newline = i;
            }
            while (run_end < units.size() && is_space(units[run_end].codepoint)) {
                if (is_newline(units[run_end].codepoint)) {
                    last_newline = run_end;
                }
                ++run_end;
            }
            const size_t piece_end = last_newline == std::numeric_limits<size_t>::max()
                    ? run_end
                    : last_newline + 1;
            add_piece(i, piece_end);
            i = piece_end;
            continue;
        }

        add_piece(i, i + 1);
        ++i;
    }

    return pieces;
}

std::vector<std::string> Qwen3Tokenizer::bpe(
        const std::string& piece) const {
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (unsigned char byte : piece) {
        symbols.push_back(byte_encoder_[byte]);
    }

    while (symbols.size() >= 2) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        MergePair best_pair;
        bool found = false;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            MergePair pair{symbols[i], symbols[i + 1]};
            const auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pair = std::move(pair);
                found = true;
            }
        }
        if (!found) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size()
                    && symbols[i] == best_pair.first
                    && symbols[i + 1] == best_pair.second) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(std::move(symbols[i]));
                ++i;
            }
        }
        symbols = std::move(merged);
    }

    return symbols;
}

std::vector<int32_t> Qwen3Tokenizer::encode_normal(
        const std::string& text) const {
    std::vector<int32_t> result;
    for (const std::string& piece : pretokenize(text)) {
        const std::vector<std::string> symbols = bpe(piece);
        for (const std::string& symbol : symbols) {
            const auto it = token_to_id_.find(symbol);
            if (it == token_to_id_.end()) {
                throw std::runtime_error(
                        "Qwen3 tokenizer vocabulary is missing byte/BPE token");
            }
            result.push_back(it->second);
        }
    }
    return result;
}

std::vector<int32_t> Qwen3Tokenizer::encode(
        const std::string& text,
        const Qwen3EncodeOptions& options) const {
    if (!ready_) {
        throw std::runtime_error("Qwen3 tokenizer is not loaded");
    }

    std::vector<int32_t> result;
    const bool add_bos = options.add_bos || add_bos_token_;
    if (add_bos && bos_token_id_ >= 0) {
        result.push_back(bos_token_id_);
    }

    if (!options.parse_special || atomic_tokens_by_length_.empty()) {
        std::vector<int32_t> normal = encode_normal(text);
        result.insert(result.end(), normal.begin(), normal.end());
    } else {
        size_t normal_begin = 0;
        size_t offset = 0;
        while (offset < text.size()) {
            int32_t special_id = -1;
            size_t special_length = 0;
            for (const auto& entry : atomic_tokens_by_length_) {
                if (starts_with_at(text, offset, entry.first)) {
                    special_id = entry.second;
                    special_length = entry.first.size();
                    break;
                }
            }

            if (special_id < 0) {
                ++offset;
                continue;
            }

            if (normal_begin < offset) {
                std::vector<int32_t> normal = encode_normal(
                        text.substr(normal_begin, offset - normal_begin));
                result.insert(result.end(), normal.begin(), normal.end());
            }
            result.push_back(special_id);
            offset += special_length;
            normal_begin = offset;
        }
        if (normal_begin < text.size()) {
            std::vector<int32_t> normal = encode_normal(text.substr(normal_begin));
            result.insert(result.end(), normal.begin(), normal.end());
        }
    }

    if (options.add_eos && eos_token_id_ >= 0) {
        result.push_back(eos_token_id_);
    }
    return result;
}

std::vector<int32_t> Qwen3Tokenizer::encode(
        const std::string& text,
        bool add_bos,
        bool add_eos) const {
    Qwen3EncodeOptions options;
    options.parse_special = true;
    options.add_bos = add_bos;
    options.add_eos = add_eos;
    return encode(text, options);
}

std::string Qwen3Tokenizer::decode(
        const std::vector<int32_t>& token_ids,
        bool skip_special) const {
    if (!ready_) {
        throw std::runtime_error("Qwen3 tokenizer is not loaded");
    }

    std::string result;
    for (int32_t id : token_ids) {
        if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) {
            throw std::out_of_range("token id outside Qwen3 vocabulary");
        }
        if (skip_special && is_control_token(id)) {
            continue;
        }

        const std::string& token = id_to_token_[id];
        if (is_atomic_token(id)) {
            result += token;
            continue;
        }

        for (const Utf8Unit& unit : utf8_units(token)) {
            const auto byte = byte_decoder_.find(unit.codepoint);
            if (byte != byte_decoder_.end()) {
                result.push_back(static_cast<char>(byte->second));
            } else {
                result.append(token, unit.begin, unit.end - unit.begin);
            }
        }
    }
    return result;
}

std::string Qwen3Tokenizer::format_chat_prompt(
        const std::string& user_text,
        const std::string& system_text,
        bool enable_thinking) const {
    std::string prompt;
    if (!system_text.empty()) {
        prompt += "<|im_start|>system\n";
        prompt += system_text;
        prompt += "<|im_end|>\n";
    }
    prompt += "<|im_start|>user\n";
    prompt += user_text;
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    if (!enable_thinking) {
        prompt += "<think>\n\n</think>\n\n";
    }
    return prompt;
}
