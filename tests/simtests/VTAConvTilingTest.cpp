// Checks convolution_wo_tr on the VTA simulator against a reference
// convolution, for output sizes that need edge tiles smaller than the minimum
// tile (MIN_H_SIZE x MIN_W_SIZE in VTABundle.cpp). Such tiles are padded up
// with dummy rows/columns; the cases below come from SSD300-VGG16 (19x19 and
// 5x5 feature maps), which exercises those paths unlike the ResNet models.
#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "vta/runtime.h"
#include "vta/vtalib/include/Bundle/VTABundle.h"

namespace {

struct ConvCase {
  int H, W, C, KN, K, pad, stride;
  bool relu;
  uint8_t shift;
};

VTACommandHandle cmdHandle() {
  // One handle for the whole test binary: the bundle library caches micro-op
  // kernels per configuration and those must stay valid.
  static VTACommandHandle handle = VTATLSCommandHandle();
  return handle;
}

int8_t clampInt8(int32_t v) {
  return int8_t(v > 127 ? 127 : (v < -128 ? -128 : v));
}

void runCase(const ConvCase &c) {
  const int outH = (c.H + 2 * c.pad - c.K) / c.stride + 1;
  const int outW = (c.W + 2 * c.pad - c.K) / c.stride + 1;
  std::mt19937 rng(c.H * 1000 + c.C + c.KN + c.stride);
  std::uniform_int_distribution<int> small(-8, 8);

  // NHWC data and [KN, KH, KW, C] filter, as Glow sees them.
  std::vector<int8_t> in(c.H * c.W * c.C), filter(c.KN * c.K * c.K * c.C);
  std::vector<int32_t> bias(c.KN);
  for (auto &v : in) v = int8_t(small(rng));
  for (auto &v : filter) v = int8_t(small(rng));
  for (auto &v : bias) v = small(rng) * 64;

  // Reference: int32 accumulate, add bias, arithmetic shift, (relu), clamp.
  std::vector<int8_t> ref(outH * outW * c.KN);
  for (int oy = 0; oy < outH; oy++)
    for (int ox = 0; ox < outW; ox++)
      for (int k = 0; k < c.KN; k++) {
        int32_t acc = bias[k];
        for (int ky = 0; ky < c.K; ky++)
          for (int kx = 0; kx < c.K; kx++) {
            int iy = oy * c.stride - c.pad + ky, ix = ox * c.stride - c.pad + kx;
            if (iy < 0 || ix < 0 || iy >= c.H || ix >= c.W) continue;
            for (int ch = 0; ch < c.C; ch++)
              acc += int32_t(in[(iy * c.W + ix) * c.C + ch]) *
                     filter[((k * c.K + ky) * c.K + kx) * c.C + ch];
          }
        acc >>= c.shift;
        if (c.relu && acc < 0) acc = 0;
        ref[(oy * outW + ox) * c.KN + k] = clampInt8(acc);
      }

  // VTA layouts: data [C/16][H][W][16], filter [KN/16][C/16][KH][KW][16][16].
  std::vector<int8_t> inV(in.size()), filterV(filter.size());
  for (int y = 0; y < c.H; y++)
    for (int x = 0; x < c.W; x++)
      for (int ch = 0; ch < c.C; ch++)
        inV[((ch / 16 * c.H + y) * c.W + x) * 16 + ch % 16] =
            in[(y * c.W + x) * c.C + ch];
  for (int k = 0; k < c.KN; k++)
    for (int ky = 0; ky < c.K; ky++)
      for (int kx = 0; kx < c.K; kx++)
        for (int ch = 0; ch < c.C; ch++)
          filterV[((((k / 16 * (c.C / 16) + ch / 16) * c.K + ky) * c.K + kx) *
                       16 + k % 16) * 16 + ch % 16] =
              filter[((k * c.K + ky) * c.K + kx) * c.C + ch];

  void *inBuf = VTABufferAlloc(inV.size());
  void *filterBuf = VTABufferAlloc(filterV.size());
  void *biasBuf = VTABufferAlloc(bias.size() * 4);
  void *outBuf = VTABufferAlloc(ref.size());
  VTABufferCopy(inV.data(), 0, inBuf, 0, inV.size(), 1);
  VTABufferCopy(filterV.data(), 0, filterBuf, 0, filterV.size(), 1);
  VTABufferCopy(bias.data(), 0, biasBuf, 0, bias.size() * 4, 1);

  ASSERT_EQ(0, convolution_wo_tr((int8_t *)inBuf, (int8_t *)filterBuf,
                                 (int32_t *)biasBuf, (int8_t *)outBuf, 1, c.H,
                                 c.W, c.C, c.KN, c.K, c.K, c.pad, c.stride,
                                 c.relu, true, c.shift, outH, outW,
                                 cmdHandle()));

  std::vector<int8_t> outV(ref.size());
  VTABufferCopy(outBuf, 0, outV.data(), 0, outV.size(), 2);
  VTABufferFree(inBuf);
  VTABufferFree(filterBuf);
  VTABufferFree(biasBuf);
  VTABufferFree(outBuf);

  int mismatches = 0;
  for (int y = 0; y < outH; y++)
    for (int x = 0; x < outW; x++)
      for (int k = 0; k < c.KN; k++) {
        int8_t got = outV[((k / 16 * outH + y) * outW + x) * 16 + k % 16];
        int8_t want = ref[(y * outW + x) * c.KN + k];
        if (got != want && mismatches++ < 5) {
          ADD_FAILURE() << "out[" << y << "][" << x << "][" << k
                        << "] = " << int(got) << ", expected " << int(want);
        }
      }
  EXPECT_EQ(0, mismatches);
}

} // namespace

// Fits one 14x14 tile: no padded tiles (the path the ResNet models use).
TEST(VTAConvTilingTest, FullTile) { runCase({14, 14, 32, 32, 3, 1, 1, true, 6}); }

// 19 = 14 + 5: a 5-wide edge tile, padded to 7, with two filter groups
// (SSD extra.0 convs).
TEST(VTAConvTilingTest, NarrowEdgeTile) {
  runCase({19, 19, 64, 512, 3, 1, 1, true, 6});
}

// The padded 7x7 tile must also bound the filters per ACC load (SSD conv7).
TEST(VTAConvTilingTest, NarrowEdgeTileManyFilters) {
  runCase({19, 19, 32, 1024, 1, 0, 1, false, 4});
}

// Whole output smaller than the minimum tile, strided (SSD extra.2).
TEST(VTAConvTilingTest, SmallStridedOutput) {
  runCase({10, 10, 32, 256, 3, 1, 2, true, 6});
}

// 3x3 output from a 5x5 input without padding (SSD extra.3).
TEST(VTAConvTilingTest, SmallOutputNoPad) {
  runCase({5, 5, 32, 64, 3, 0, 1, true, 6});
}

// The SSD300-VGG16 extra layers after the 10x10 map, in network order.
TEST(VTAConvTilingTest, SsdExtraLayers) {
  runCase({10, 10, 512, 128, 1, 0, 1, true, 6});
  runCase({10, 10, 128, 256, 3, 1, 2, true, 6});
  runCase({5, 5, 256, 128, 1, 0, 1, true, 6});
  runCase({5, 5, 128, 256, 3, 0, 1, true, 6});
  runCase({3, 3, 256, 128, 1, 0, 1, true, 6});
  runCase({3, 3, 128, 256, 3, 0, 1, true, 6});
}

// Every VTA convolution of SSD300-VGG16, in network order, on one command
// handle. The micro-op kernels no longer fit the micro-op SRAM, so this covers
// the wrap-around of the micro-op cache (lib/simulator/runtime.cc UopQueue).
// Takes about 15 s on the simulator.
TEST(VTAConvTilingTest, SsdAllLayers) {
  const ConvCase cases[] = {
    {300, 300, 64, 64, 3, 1, 1, true, 6},
    {150, 150, 64, 128, 3, 1, 1, true, 6},
    {150, 150, 128, 128, 3, 1, 1, true, 6},
    {75, 75, 128, 256, 3, 1, 1, true, 6},
    {75, 75, 256, 256, 3, 1, 1, true, 6},
    {75, 75, 256, 256, 3, 1, 1, true, 6},
    {38, 38, 256, 512, 3, 1, 1, true, 6},
    {38, 38, 512, 512, 3, 1, 1, true, 6},
    {38, 38, 512, 512, 3, 1, 1, true, 6},
    {38, 38, 512, 16, 3, 1, 1, false, 6},
    {19, 19, 512, 512, 3, 1, 1, true, 6},
    {19, 19, 512, 512, 3, 1, 1, true, 6},
    {19, 19, 512, 512, 3, 1, 1, true, 6},
    {19, 19, 1024, 1024, 1, 0, 1, true, 6},
    {19, 19, 1024, 256, 1, 0, 1, true, 6},
    {19, 19, 256, 512, 3, 1, 2, true, 6},
    {10, 10, 512, 128, 1, 0, 1, true, 6},
    {10, 10, 128, 256, 3, 1, 2, true, 6},
    {5, 5, 256, 128, 1, 0, 1, true, 6},
    {5, 5, 128, 256, 3, 0, 1, true, 6},
    {3, 3, 256, 128, 1, 0, 1, true, 6},
    {3, 3, 128, 256, 3, 0, 1, true, 6},
    {1, 1, 256, 16, 1, 0, 1, false, 6},
    {3, 3, 256, 16, 3, 1, 1, false, 6}};
  int i = 0;
  for (const auto &c : cases) {
    SCOPED_TRACE(i++);
    runCase(c);
  }
}
