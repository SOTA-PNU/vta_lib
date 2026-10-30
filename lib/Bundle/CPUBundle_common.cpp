// Layout-independent CPU operators shared by the generic and aarch64 bundles.
//
// The VTA bundle saver calls these for Glow instructions that have no VTA
// counterpart (e.g. the L2 normalization and the head concatenation of SSD).
// Unlike the int8-only helpers in CPUBundle_*.cpp they take the tensor rank
// (up to kMaxRank) and, where the data is only moved, the element size, so
// the same function serves int8 and float tensors.
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vta/vtalib/include/Bundle/VTABundle.h"

namespace {

constexpr unsigned_t kMaxRank = 6;

/// Row-major strides (in elements) of a tensor with dims \p dims.
void computeStrides(const dim_t *dims, unsigned_t rank, dim_t *strides) {
  dim_t s = 1;
  for (int i = int(rank) - 1; i >= 0; --i) {
    strides[i] = s;
    s *= dims[i];
  }
}

dim_t numElements(const dim_t *dims, unsigned_t rank) {
  dim_t n = 1;
  for (unsigned_t i = 0; i < rank; ++i) {
    n *= dims[i];
  }
  return n;
}

/// Advances the multi-index \p idx over \p dims; returns false after the last.
bool nextIndex(dim_t *idx, const dim_t *dims, unsigned_t rank) {
  for (int i = int(rank) - 1; i >= 0; --i) {
    if (++idx[i] < dims[i]) {
      return true;
    }
    idx[i] = 0;
  }
  return false;
}

} // namespace

int transpose_nd(const int8_t *input, int8_t *output, const dim_t *inDims,
                 const unsigned_t *shuffle, unsigned_t rank,
                 unsigned_t elemSize) {
  if (rank == 0 || rank > kMaxRank) {
    return -1;
  }
  // Glow semantics: outDims[i] = inDims[shuffle[i]].
  dim_t outDims[kMaxRank], inStrides[kMaxRank], outStrides[kMaxRank];
  for (unsigned_t i = 0; i < rank; ++i) {
    outDims[i] = inDims[shuffle[i]];
  }
  computeStrides(inDims, rank, inStrides);
  computeStrides(outDims, rank, outStrides);
  // Walk the output in order; the matching input element of output index
  // idx is at sum(idx[i] * inStrides[shuffle[i]]).
  dim_t srcStride[kMaxRank];
  for (unsigned_t i = 0; i < rank; ++i) {
    srcStride[i] = inStrides[shuffle[i]];
  }
  if (numElements(outDims, rank) == 0) {
    return 0;
  }
  dim_t idx[kMaxRank] = {0};
  dim_t dst = 0;
  do {
    dim_t src = 0;
    for (unsigned_t i = 0; i < rank; ++i) {
      src += idx[i] * srcStride[i];
    }
    std::memcpy(output + dst * elemSize, input + src * elemSize, elemSize);
    ++dst;
  } while (nextIndex(idx, outDims, rank));
  return 0;
}

int insert_tensor(int8_t *output, const int8_t *input, const dim_t *outDims,
                  const dim_t *inDims, const dim_t *offsets, unsigned_t rank,
                  unsigned_t elemSize, dim_t count, unsigned_t axis) {
  if (rank == 0 || rank > kMaxRank || axis >= rank) {
    return -1;
  }
  // Glow semantics: copy i of the input goes to offsets + i * inDims[axis]
  // along axis.
  dim_t outStrides[kMaxRank];
  computeStrides(outDims, rank, outStrides);
  // Copy whole innermost rows at a time.
  const dim_t row = inDims[rank - 1] * elemSize;
  if (numElements(inDims, rank) == 0) {
    return 0;
  }
  for (dim_t c = 0; c < count; ++c) {
    dim_t base = 0;
    for (unsigned_t i = 0; i < rank; ++i) {
      dim_t off = offsets[i] + (i == axis ? c * inDims[axis] : 0);
      base += off * outStrides[i];
    }
    dim_t idx[kMaxRank] = {0};
    const int8_t *src = input;
    do {
      dim_t dst = base;
      for (unsigned_t i = 0; i + 1 < rank; ++i) {
        dst += idx[i] * outStrides[i];
      }
      std::memcpy(output + dst * elemSize, src, row);
      src += row;
    } while (nextIndex(idx, inDims, rank - 1));
  }
  return 0;
}

int splat_f32(int8_t *output, dim_t size, float value) {
  float *out = (float *)output;
  for (dim_t i = 0; i < size; ++i) {
    out[i] = value;
  }
  return 0;
}

int elemmul_f32(int8_t *input0, int8_t *input1, int8_t *output, dim_t size) {
  const float *l = (const float *)input0;
  const float *r = (const float *)input1;
  float *out = (float *)output;
  for (dim_t i = 0; i < size; ++i) {
    out[i] = l[i] * r[i];
  }
  return 0;
}

int elemmax_f32(int8_t *input0, int8_t *input1, int8_t *output, dim_t size) {
  const float *l = (const float *)input0;
  const float *r = (const float *)input1;
  float *out = (float *)output;
  for (dim_t i = 0; i < size; ++i) {
    out[i] = l[i] < r[i] ? r[i] : l[i];
  }
  return 0;
}

int elempow_f32(int8_t *input0, int8_t *input1, int8_t *output, dim_t size) {
  const float *l = (const float *)input0;
  const float *r = (const float *)input1;
  float *out = (float *)output;
  for (dim_t i = 0; i < size; ++i) {
    out[i] = std::pow(l[i], r[i]);
  }
  return 0;
}

int batched_reduce_add_f32(int8_t *input, int8_t *output, const dim_t *inDims,
                           unsigned_t rank, unsigned_t axis) {
  if (rank == 0 || rank > kMaxRank || axis >= rank) {
    return -1;
  }
  // View the input as [outer, axis, inner] and sum over the middle.
  dim_t outer = 1, inner = 1;
  for (unsigned_t i = 0; i < axis; ++i) {
    outer *= inDims[i];
  }
  for (unsigned_t i = axis + 1; i < rank; ++i) {
    inner *= inDims[i];
  }
  const dim_t n = inDims[axis];
  const float *in = (const float *)input;
  float *out = (float *)output;
  for (dim_t o = 0; o < outer; ++o) {
    float *dst = out + o * inner;
    for (dim_t j = 0; j < inner; ++j) {
      dst[j] = 0;
    }
    for (dim_t k = 0; k < n; ++k) {
      const float *src = in + (o * n + k) * inner;
      for (dim_t j = 0; j < inner; ++j) {
        dst[j] += src[j];
      }
    }
  }
  return 0;
}
