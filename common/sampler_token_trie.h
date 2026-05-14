/*
 * Token-trie sampler — C/C++ interface for constrained token generation.
 *
 * This header declares the public entry point for the token-trie sampler
 * used by the elizaOS inference stack. The sampler consumes a serialised
 * TokenTreePayload (produced by the TS side at
 * `packages/ui/src/services/local-inference/token-tree.ts`) and restricts
 * the candidate set to only the tokens that are legal continuations of the
 * current trie position.
 *
 * See `plugins/plugin-local-inference/native/docs/TOKEN_TRIE_SAMPLER.md`
 * for the full design, payload schema, and lifecycle contract.
 *
 * The sampler integrates into a llama_sampler_chain the same way any other
 * llama.cpp sampler does. After being added with llama_sampler_chain_add,
 * the chain owns the sampler and frees it when the chain is freed.
 */

#pragma once

#include "llama.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialise a token-trie constraint sampler.
 *
 * @param vocab               The model's vocabulary (from
 *                            llama_model_get_vocab()). Borrowed; the
 *                            sampler does not take ownership.
 * @param trie_descriptor_json
 *                            Serialised TokenTreePayload (JSON). The
 *                            payload schema is:
 *                            {
 *                              "modelId": "<id>",
 *                              "descriptors": [
 *                                {
 *                                  "path": "action" | "parameters.x" | ...,
 *                                  "leaves": [
 *                                    { "name": "X", "tokens": [n, n, ...] },
 *                                    ...
 *                                  ]
 *                                },
 *                                ...
 *                              ]
 *                            }
 *                            The string is consumed (parsed + cached) by
 *                            the sampler. The caller may free it on return.
 * @param mode                0 = argmax-greedy. When exactly one valid
 *                              next-token exists the sampler collapses the
 *                              candidate set to that single token so the
 *                              downstream selector skips the forward pass.
 *                              When multiple valid tokens exist the
 *                              sampler masks the rest to -INFINITY and lets
 *                              the chain pick argmax across the survivors.
 *                            1 = sampled-from-filtered. The sampler masks
 *                              invalid tokens to -INFINITY and leaves the
 *                              survivors' logits untouched so a downstream
 *                              temperature / top-P / dist sampler runs over
 *                              just the valid set.
 *
 * Returns a llama_sampler* on success, or NULL on:
 *   - JSON parse error,
 *   - empty leaf set (no descriptors carry any token sequence),
 *   - memory allocation failure.
 *
 * Caller transfers ownership to the llama_sampler_chain via
 * llama_sampler_chain_add. The chain frees the sampler when it is itself
 * freed; callers must NOT call llama_sampler_free on the returned pointer
 * after handing it to the chain.
 */
struct llama_sampler * llama_sampler_init_token_trie(
    const struct llama_vocab * vocab,
    const char * trie_descriptor_json,
    int mode);

#ifdef __cplusplus
}
#endif
