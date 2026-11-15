// tensormark/tmtok.h — one tokenizer surface for every Llama-family CLI.
//
// Each CLI used to assemble its own tokenizer from the same pieces: llama_chat
// had a full Tokenizer struct (SentencePiece or byte-level BPE plus the
// piece_bytes/decode_delta/is_control plumbing the streaming decode and the
// grammar mask need), eval_ppl_llama/spec_bench/align_probe hand-rolled just an
// Encoder with their own path arithmetic. open_for_model() is the single
// construction point: it decides the format from the path's SUFFIX (the same
// rule llama_chat used) and wires the same helpers, so a probe and the chat CLI
// cannot drift apart on how a tokenizer maps a model directory.
//
// Path conventions, unchanged:
//   * "<x>.vocab.json"  -> byte-level BPE (tmbpe), with "<x>.merges.txt" and the
//     optional "<x>.special.json" sidecar;
//   * anything else     -> SentencePiece (tmspm), typically "tokenizer.model".
#pragma once

#include "bpe.h"
#include "sp_tokenizer.h"
#include <string>

namespace tmtok {

// The union tokenizer the chat CLI uses. `bpe` selects the encoder; both halves
// expose the same encode/decode/bos/eos surface.
struct Tokenizer {
    bool bpe = false;
    tmspm::Tokenizer spm;
    tmbpe::Encoder enc;
    [[nodiscard]] int bos_id() const { return bpe ? enc.bos_id : spm.bos_id; }
    [[nodiscard]] int eos_id() const { return bpe ? enc.eos_id : spm.eos_id; }
    [[nodiscard]] std::vector<int> encode(const std::string& s) const {
        return bpe ? enc.encode_qwen(s) : spm.encode(s);
    }
    [[nodiscard]] std::string decode(const std::vector<int>& ids) const {
        return bpe ? enc.decode(ids) : spm.decode(ids);
    }
    // The bytes this id contributes to a running decode (from the per-id table
    // built at load — see the respective tokenizer).
    [[nodiscard]] const std::string& piece_bytes(int id) const {
        return bpe ? enc.piece_bytes(id) : spm.piece_bytes(id);
    }
    [[nodiscard]] std::string decode_delta(int id, bool first_piece) const {
        return bpe ? enc.decode_delta(id, first_piece) : spm.decode_delta(id, first_piece);
    }
    [[nodiscard]] bool is_control(int id) const {
        return bpe ? enc.is_control(id) : spm.is_control(id);
    }
};

// Loads the tokenizer for `tokm` (a path). `use_qwen_split` enables the
// Qwen2/LLaMA-3 pretokenizer on the byte-level side — the chat CLI and every
// probe want it; a caller reproducing raw GPT-2 BPE would pass false.
[[nodiscard]] inline Tokenizer open_for_model(const std::string& tokm,
                                              bool use_qwen_split = true) {
    Tokenizer tok;
    if (tokm.size() > 11 && tokm.compare(tokm.size() - 11, 11, ".vocab.json") == 0) {
        // byte-level BPE: vocab.json + merges.txt + special.json sidecar
        const std::string base = tokm.substr(0, tokm.size() - std::string(".vocab.json").size());
        tok.bpe = true;
        tok.enc = tmbpe::Encoder::load(tokm, base + ".merges.txt");
        tok.enc.load_special(base + ".special.json");
        (void)use_qwen_split;   // encode_qwen is always the entry on this side
    } else {
        tok.spm.load(tokm);
    }
    return tok;
}

}  // namespace tmtok
