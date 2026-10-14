// SPDX-License-Identifier: MIT

/**
 * Shares the loaded Kokoro tensor context across the owned predictor and
 * decoder translation units. Tensor-name compatibility stays at this boundary
 * so every stage consumes the same canonical or published schema.
 */

#pragma once

#include <string>

struct ggml_context;
struct ggml_tensor;

namespace eliza_kokoro {

struct kokoro_model;

ggml_context * kokoro_model_ggml_ctx(const kokoro_model * model);
ggml_tensor * kokoro_find_tensor_compat(ggml_context * ctx, const std::string & name);

}
