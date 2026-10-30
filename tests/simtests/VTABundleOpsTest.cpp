// Tests for the layout-independent operators in CPUBundle_common.cpp. The
// expected values are computed with straightforward per-element loops.
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "vta/vtalib/include/Bundle/VTABundle.h"

namespace {

std::vector<float> iota(size_t n, float start = 0) {
  std::vector<float> v(n);
  std::iota(v.begin(), v.end(), start);
  return v;
}

int8_t *bytes(std::vector<float> &v) { return (int8_t *)v.data(); }

} // namespace

TEST(VTABundleOpsTest, TransposeNd5dFloat) {
  // [1, 16, 3, 2] viewed as [1, 4, 4, 3, 2] permuted (0, 3, 4, 1, 2), as in
  // the SSD head reshape.
  const dim_t in[5] = {1, 4, 4, 3, 2};
  const unsigned_t shuffle[5] = {0, 3, 4, 1, 2};
  std::vector<float> src = iota(96), dst(96, -1);
  ASSERT_EQ(0, transpose_nd(bytes(src), bytes(dst), in, shuffle, 5, 4));
  // out[0][h][w][a][k] = in[0][a][k][h][w]
  for (dim_t h = 0; h < 3; ++h)
    for (dim_t w = 0; w < 2; ++w)
      for (dim_t a = 0; a < 4; ++a)
        for (dim_t k = 0; k < 4; ++k)
          EXPECT_EQ(src[((a * 4 + k) * 3 + h) * 2 + w],
                    dst[((h * 2 + w) * 4 + a) * 4 + k]);
}

TEST(VTABundleOpsTest, TransposeNdMatches4dInt8) {
  const dim_t in[4] = {2, 3, 4, 5};
  const unsigned_t shuffle[4] = {0, 2, 3, 1};
  std::vector<int8_t> src(120), ref(120), dst(120);
  for (size_t i = 0; i < src.size(); ++i) src[i] = int8_t(i - 60);
  ASSERT_EQ(0, transpose(src.data(), ref.data(), 2, 3, 4, 5, 2, 4, 5, 3, 0, 2,
                         3, 1));
  ASSERT_EQ(0, transpose_nd(src.data(), dst.data(), in, shuffle, 4, 1));
  EXPECT_EQ(ref, dst);
}

TEST(VTABundleOpsTest, TransposeNd3d) {
  const dim_t in[3] = {2, 3, 4};
  const unsigned_t shuffle[3] = {2, 0, 1};
  std::vector<float> src = iota(24), dst(24);
  ASSERT_EQ(0, transpose_nd(bytes(src), bytes(dst), in, shuffle, 3, 4));
  for (dim_t i = 0; i < 2; ++i)
    for (dim_t j = 0; j < 3; ++j)
      for (dim_t k = 0; k < 4; ++k)
        EXPECT_EQ(src[(i * 3 + j) * 4 + k], dst[(k * 2 + i) * 3 + j]);
}

TEST(VTABundleOpsTest, InsertTensorConcat) {
  // Concat [1, 5, 3] and [1, 5, 2] along axis 2 into [1, 5, 5].
  const dim_t out[3] = {1, 5, 5}, a[3] = {1, 5, 3}, b[3] = {1, 5, 2};
  const dim_t offA[3] = {0, 0, 0}, offB[3] = {0, 0, 3};
  std::vector<float> va = iota(15), vb = iota(10, 100), dst(25, -1);
  ASSERT_EQ(0, insert_tensor(bytes(dst), bytes(va), out, a, offA, 3, 4, 1, 2));
  ASSERT_EQ(0, insert_tensor(bytes(dst), bytes(vb), out, b, offB, 3, 4, 1, 2));
  for (dim_t r = 0; r < 5; ++r)
    for (dim_t c = 0; c < 5; ++c)
      EXPECT_EQ(c < 3 ? va[r * 3 + c] : vb[r * 2 + c - 3], dst[r * 5 + c]);
}

TEST(VTABundleOpsTest, InsertTensorPadInt8) {
  // Zero-pad bottom/right of a [1, 3, 3, 2] NHWC tensor to [1, 4, 4, 2].
  const dim_t out[4] = {1, 4, 4, 2}, in[4] = {1, 3, 3, 2}, off[4] = {0, 0, 0, 0};
  std::vector<int8_t> src(18), dst(32, 0);
  for (size_t i = 0; i < src.size(); ++i) src[i] = int8_t(i + 1);
  ASSERT_EQ(0, insert_tensor(dst.data(), src.data(), out, in, off, 4, 1, 1, 0));
  for (dim_t h = 0; h < 4; ++h)
    for (dim_t w = 0; w < 4; ++w)
      for (dim_t c = 0; c < 2; ++c)
        EXPECT_EQ(h < 3 && w < 3 ? src[(h * 3 + w) * 2 + c] : 0,
                  dst[(h * 4 + w) * 2 + c]);
}

TEST(VTABundleOpsTest, InsertTensorTile) {
  // Broadcast [1, 4, 1] to [1, 4, 3] with count 3 along axis 2.
  const dim_t out[3] = {1, 4, 3}, in[3] = {1, 4, 1}, off[3] = {0, 0, 0};
  std::vector<float> src = iota(4, 1), dst(12, 0);
  ASSERT_EQ(0, insert_tensor(bytes(dst), bytes(src), out, in, off, 3, 4, 3, 2));
  for (dim_t r = 0; r < 4; ++r)
    for (dim_t c = 0; c < 3; ++c)
      EXPECT_EQ(src[r], dst[r * 3 + c]);
}

TEST(VTABundleOpsTest, ElementwiseFloat) {
  std::vector<float> l = {-2, 0.5, 4, 9}, r = {3, 0.25, 0.5, 0.5}, o(4);
  ASSERT_EQ(0, elemmul_f32(bytes(l), bytes(r), bytes(o), 4));
  EXPECT_EQ((std::vector<float>{-6, 0.125, 2, 4.5}), o);
  ASSERT_EQ(0, elemmax_f32(bytes(l), bytes(r), bytes(o), 4));
  EXPECT_EQ((std::vector<float>{3, 0.5, 4, 9}), o);
  std::vector<float> p = {4, 9, 0, 2}, e = {0.5, 0.5, 0.5, 3};
  ASSERT_EQ(0, elempow_f32(bytes(p), bytes(e), bytes(o), 4));
  EXPECT_EQ((std::vector<float>{2, 3, 0, 8}), o);
  ASSERT_EQ(0, splat_f32(bytes(o), 4, 1.5f));
  EXPECT_EQ((std::vector<float>{1.5, 1.5, 1.5, 1.5}), o);
}

TEST(VTABundleOpsTest, BatchedReduceAdd) {
  const dim_t in[4] = {2, 3, 2, 2};
  std::vector<float> src = iota(24);
  for (unsigned_t axis = 0; axis < 4; ++axis) {
    std::vector<float> dst(24 / in[axis], -1), ref(dst.size(), 0);
    ASSERT_EQ(0, batched_reduce_add_f32(bytes(src), bytes(dst), in, 4, axis));
    dim_t idx[4];
    for (idx[0] = 0; idx[0] < 2; ++idx[0])
      for (idx[1] = 0; idx[1] < 3; ++idx[1])
        for (idx[2] = 0; idx[2] < 2; ++idx[2])
          for (idx[3] = 0; idx[3] < 2; ++idx[3]) {
            dim_t o = 0;
            for (unsigned_t i = 0; i < 4; ++i)
              if (i != axis) o = o * in[i] + idx[i];
            ref[o] += src[((idx[0] * 3 + idx[1]) * 2 + idx[2]) * 2 + idx[3]];
          }
    EXPECT_EQ(ref, dst) << "axis " << axis;
  }
}

TEST(VTABundleOpsTest, BadRank) {
  const dim_t d[7] = {1, 1, 1, 1, 1, 1, 1};
  const unsigned_t s[7] = {0, 1, 2, 3, 4, 5, 6};
  int8_t x = 0, y = 0;
  EXPECT_NE(0, transpose_nd(&x, &y, d, s, 7, 1));
  EXPECT_NE(0, insert_tensor(&y, &x, d, d, d, 3, 1, 1, 3));
}
