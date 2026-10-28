#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// a soft prompt file (examples/soft-prompt): [before tokens] [K soft embeddings] [after tokens] in front of every
// prompt, optionally with a tuned recurrent state that replaces the one after the soft embeddings.
// The prefix is evaluated once at load time and its sequence state is kept; in the prompt cache it stands as `tokens`
// (the K embeddings as placeholder tokens), and a slot that has to process those from scratch gets the state instead.
struct server_soft_prompt {
    std::vector<llama_token> tokens;
    std::vector<uint8_t>     state;

    bool empty() const { return tokens.empty(); }

    // evaluates the prefix on sequence 0 (which must be empty) and keeps its state; throws on error
    void init(llama_context * ctx, const std::string & path);

    // replaces the memory of seq_id by the prefix
    bool restore(llama_context * ctx, llama_seq_id seq_id) const;

    // the tokens start with the prefix
    bool is_prefix_of(const std::vector<llama_token> & toks) const;
};
