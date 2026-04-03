// Copyright © 2023-2024 Apple Inc.
#pragma once

#include "mlx/io.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

extern "C" {
#include <gguflib.h>
}

// Maximum number of tensor dimensions supported by the GGUF format.
// Mirrors GGUF_TENSOR_MAX_DIM from gguflib.h. Override at compile time
// with -DMLX_GGUF_MAX_DIMS=<value> if the upstream format changes.
#ifndef MLX_GGUF_MAX_DIMS
#define MLX_GGUF_MAX_DIMS GGUF_TENSOR_MAX_DIM
#endif

namespace mlx::core {

Shape get_shape(const gguf_tensor& tensor);
void gguf_load_quantized(
    std::unordered_map<std::string, array>& a,
    const gguf_tensor& tensor);

} // namespace mlx::core
