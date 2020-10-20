#include <iostream>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cassert>
#include "llvm/ADT/ArrayRef.h"
#include "glow/Base/Tensor.h"
#include "glow/Quantization/Base/Base.h"

#include <arm_neon.h>
#include "vta/vtalib/include/Bundle/VTABundle.h"

#define DEBUG_INFO  0

using namespace std;
using namespace glow;

inline void nhwc_to_nchw_s8(
  const int8_t* const in,  // Input data in NHWC form
  int8_t* const out,       // Output data in NCHW form
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels
)
{
  // Fill in stride values
  int in_col_stride = n_channels;
  int in_row_stride = n_cols * in_col_stride;
  int in_batch_stride = n_rows * in_row_stride;

  int out_row_stride = n_cols;
  int out_channel_stride = n_rows * out_row_stride;
  int out_batch_stride = n_channels * out_channel_stride;

  // Perform the re-ordering
  // For every batch
  for (int n = 0; n < n_batches; n++)
  {
    const int8_t* const in_batch = in + n*in_batch_stride;
    int8_t* const out_batch = out + n*out_batch_stride;

    // For every row
    for (int i = 0; i < n_rows; i++)
    {
      const int8_t* const in_i = in_batch + i*in_row_stride;
      int8_t* const out_i = out_batch + i*out_row_stride;

      // For every column
      for (int j = 0; j < n_cols; j++)
      {
        const int8_t* const in_j = in_i + j*in_col_stride;
        int8_t* const out_j = out_i + j;

        // For every channel
        for (int c = 0; c < n_channels; c++)
        {
          const int8_t* const in_channel = in_j + c;
          int8_t* const out_channel = out_j + c*out_channel_stride;
          *(out_channel) = *(in_channel);
        }
      }
    }
  }
}
// only channel 3. width 16���
inline void nhwc_to_nchw_s8_c3_w16n(
  const int8_t* const in,  // Input data in NHWC form
  int8_t* const out,       // Output data in NCHW form
  const int n_batches,
  const int n_rows,
  const int n_cols,
  const int n_channels
)
{
  // Fill in stride values
  int in_col_stride = n_channels;
  int in_row_stride = n_cols * in_col_stride;
  int in_batch_stride = n_rows * in_row_stride;

  int out_row_stride = n_cols;
  int out_channel_stride = n_rows * out_row_stride;
  int out_batch_stride = n_channels * out_channel_stride;

  // Perform the re-ordering
  // For every batch
  for (int n = 0; n < n_batches; n++)
  {
    const int8_t* const in_batch = in + n*in_batch_stride;
    int8_t* const out_batch = out + n*out_batch_stride;

    // For every row
    for (int i = 0; i < n_rows; i++)
    {
      const int8_t* const in_i = in_batch + i*in_row_stride;
      int8_t* const out_i = out_batch + i*out_row_stride;

      // For every column
      //assert(n_cols/16)
      int num8x16 = n_cols/16;
      int8x16x3_t intlv_rgb;
      for (int j = 0; j < num8x16; j++)
      {
        const int8_t* const in_j = in_i + j*16*in_col_stride;
        int8_t* const out_j = out_i + j*16;

        intlv_rgb=vld3q_s8(in_j);
        vst1q_s8(out_j, intlv_rgb.val[0]);
        vst1q_s8(out_j+out_channel_stride, intlv_rgb.val[1]);
        vst1q_s8(out_j+out_channel_stride*2, intlv_rgb.val[2]);


      }
    }
  }
}



// 8bit nchw_to_nhwc
inline void nchw_to_nhwc_s8(
  const int8_t* const in,
  int8_t* const out,
  const int n_batches,
  const int n_channels,
  const int n_rows,
  const int n_cols
) {

  int in_row_stride =  n_cols;
  int in_channel_stride = n_rows * in_row_stride;
  int in_batch_stride = n_channels * in_channel_stride;

  int out_col_stride = n_channels;
  int out_row_stride = n_cols * out_col_stride;
  int out_batch_stride = n_rows * out_row_stride;

  // Perform the re-ordering
  for (int n = 0; n < n_batches; n++)
  {
    const int8_t* const in_batch = in + n*in_batch_stride;
    int8_t* const out_batch = out + n*out_batch_stride;

    for (int i = 0; i < n_rows; i++)
    {
      const int8_t* const in_row = in_batch + i*in_row_stride;
      int8_t* const out_row = out_batch + i*out_row_stride;

      for (int j = 0; j < n_cols; j++)
      {
        const int8_t* const in_col = in_row + j;
        int8_t* const out_col = out_row + j*out_col_stride;

        for (int c = 0; c < n_channels; c++)
        {
          const int8_t* const in_channel = in_col + c*in_channel_stride;
          out_col[c] = *(in_channel);
        }
      }
    }
  }
}

/*
// 8bit nchw_to_nhwc
inline void nchw_to_nhwc_s8(
  const int8_t* const in,
  int8_t* const out,
  const int n_batches,
  const int n_channels,
  const int n_rows,
  const int n_cols
)
{

  int in_row_stride =  n_cols;
  int in_channel_stride = n_rows * in_row_stride;
  int in_batch_stride = n_channels * in_channel_stride;

  int out_col_stride = n_channels;
  int out_row_stride = n_cols * out_col_stride;
  int out_batch_stride = n_rows * out_row_stride;

  // Perform the re-ordering
  for (int n = 0; n < n_batches; n++)
  {
    const int8_t* const in_batch = in + n*in_batch_stride;
    int8_t* const out_batch = out + n*out_batch_stride;

    for (int i = 0; i < n_rows; i++)
    {
      const int8_t* const in_row = in_batch + i*in_row_stride;
      int8_t* const out_row = out_batch + i*out_row_stride;

      int j = 0, j_remaining = n_cols;

      //16x16 transpose�� ã������.
      //�ϴ� uint16���� ��ȯ. 8x8�� ����. �ٽ� uint8�� ��ȯ.
      for (; j_remaining >= 8; j += 8, j_remaining -= 8)
      {
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 8; c += 8, c_remaining -= 8)
        {
          // Read 16 channels worth of 16 columns, then zip to produce 16 columns
          // worth of 16 channels.
          int8x8_t channel_pixels[8];

          int8_t* in_addr0 = in_row + (c + 0)*in_channel_stride + j; // +=in_channel_stride
          int8_t* out_addr0 = out_row + (j + 0)*out_col_stride + c; // += out_col_Stride

          asm volatile(
            "vld1.32 {x0}, [%[in_channel_stride]]!\n"
            "vld1.32 {x1}, [%[in_addr0]]\n"
            "vld1.8 {q0}, [x1]!\n"  // 16�� 8bit load
            "add x1,x1,x0\n"
            "vshll.8 q4,d0,#8\n"
            "vshll.8 q5,d1,#8\n"
            "vld1.8 {q0}, [x1]!\n"  // 16�� 8bit load
            "add x1,x1,x0\n"
            "vshll.8 q6,d0,#8\n"
            "vshll.8 q7,d1,#8\n"
            "vld1.8 {q0}, [x1]!\n"  // 16�� 8bit load
            "add x1,x1,x0\n"
            "vshll.8 q8,d0,#8\n"
            "vshll.8 q9,d1,#8\n"
            "vld1.8 {q0}, [x1]!\n"  // 16�� 8bit load
            "vshll.8 q10,d0,#8\n"
            "vshll.8 q11,d1,#8\n"

            // swap antidiagnoal elem within 2x2 blocks
            "vtrn.16 q8,q9\n"
            "vtrn.16 q10,q11\n"
            "vtrn.16 q4,q5\n"
            "vtrn.16 q6,q7\n"

            // swap antidiagnoal 2x2 block within 4x4 blocks
            "vtrn.32 q8,q10\n"
            "vtrn.32 q9,q11\n"
            "vtrn.32 q4,q6\n"
            "vtrn.32 q5,q7\n"

            // swap antidiagonal 4x4 blocks within 8x8 matrix
            "vswp d9,d16\n"
            "vswp d11,d18\n"
            "vswp d13,d20\n"
            "vswp d15,d22\n"

            //store

            :
            // Outputs.

            :
            // Inputs.
            [in_channel_stride] "r"(in_channel_stride), [in_addr0] "r"(in_addr0),
            [out_col_stride] "r"(out_col_stride), [out_addr0] "r"(out_addr0)
            :
            // Clobbers.
            "cc", "memory",
            // We use these NEON registers
            "x0", "x1", 
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
            "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
            "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
            "v31");

          vst1q_s32(out_row + (j + 0)*out_col_stride + c, out_0.val[0]);
          vst1q_s32(out_row + (j + 1)*out_col_stride + c, out_0.val[1]);
          vst1q_s32(out_row + (j + 2)*out_col_stride + c, out_1.val[0]);
          vst1q_s32(out_row + (j + 3)*out_col_stride + c, out_1.val[1]);
        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 4; _j++)
          {
            const T* const in_col = in_row + j + _j;
            T* const out_col = out_row + (j + _j)*out_col_stride;
            const T* const in_channel = in_col + c*in_channel_stride;
            out_col[c] = *(in_channel);
          }
        }
      }
      for (; j_remaining >= 2; j += 2, j_remaining -= 2)
      {
        int c = 0, c_remaining = n_channels;
        for (; c_remaining >= 2; c += 2, c_remaining -= 2)
        {
          // Read 2 channels worth of 2 columns, then zip to produce 2 columns
          // worth of 2 channels.
          int32x2_t channel_pixels[2];
          channel_pixels[0] = vld1_s32(in_row + (c + 0)*in_channel_stride + j);
          channel_pixels[1] = vld1_s32(in_row + (c + 1)*in_channel_stride + j);

          const auto output = vzip_s32(channel_pixels[0], channel_pixels[1]);

          vst1_s32(out_row + (j + 0)*out_col_stride + c, output.val[0]);
          vst1_s32(out_row + (j + 1)*out_col_stride + c, output.val[1]);
        }
        for (; c_remaining; c++, c_remaining--)
        {
          for (int _j = 0; _j < 2; _j++)
          {
            const T* const in_col = in_row + j + _j;
            T* const out_col = out_row + (j + _j)*out_col_stride;
            const T* const in_channel = in_col + c*in_channel_stride;
            out_col[c] = *(in_channel);
          }
        }
      }

      for (; j_remaining; j++, j_remaining--)
      {
        const T* const in_col = in_row + j;
        T* const out_col = out_row + j*out_col_stride;

        for (int c = 0; c < n_channels; c++)
        {
          const T* const in_channel = in_col + c*in_channel_stride;
          out_col[c] = *(in_channel);
        }
      }
    }
  }
}
*/


// from google gemmlowp
// lhr 4x16n, rhr 16nx4. cell size = 4x(16xN). depth = 16n
void mm_s8_4x16n(const int8_t *lhs_ptr, const int8_t *rhs_ptr, int32_t *accum_ptr, int depth)
{
    int32x4_t acc[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            acc[i][j] = vdupq_n_s32(0);
        }
    }
    for (int d = 0; d < depth; d += 16)
    {
        int8x16_t lhs[4];
        for (int i = 0; i < 4; i++)
        {
            lhs[i] = vld1q_s8(lhs_ptr + 16 * i);
        }
        int8x16_t rhs[4];
        for (int i = 0; i < 4; i++)
        {
            rhs[i] = vld1q_s8(rhs_ptr + 16 * i);
        }
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int16x8_t local_acc =
                    vmull_s8(vget_low_s8(lhs[i]), vget_low_s8(rhs[j]));
                local_acc =
                    vmlal_s8(local_acc, vget_high_s8(lhs[i]), vget_high_s8(rhs[j]));
                acc[i][j] = vpadalq_s16(acc[i][j], local_acc);
            }
        }
        lhs_ptr += 64;
        rhs_ptr += 16 * 4;
    }
    for (int i = 0; i < 4; i++)
    {
        int32x4_t acc_2x_0 = vpaddq_s32(acc[0][i], acc[1][i]);
        int32x4_t acc_2x_1 = vpaddq_s32(acc[2][i], acc[3][i]);
        int32x4_t acc_4x = vpaddq_s32(acc_2x_0, acc_2x_1);
        int32x4_t dst_val = vld1q_s32(accum_ptr + 4 * i);
        dst_val = vaddq_s32(dst_val, acc_4x);
        vst1q_s32(accum_ptr + 4 * i, dst_val);
    }
}

void fwdConvolutionInstFloatImpl(
        Tensor *inV, Tensor *outV, Tensor *filterV, Tensor *biasV,
        llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
        llvm::ArrayRef<unsigned_t> pads, size_t group, size_t dilation) {

  auto inW = inV->getHandle<float>();
  auto outW = outV->getHandle<float>();
  auto filterW = filterV->getHandle<float>();
  auto biasW = biasV->getHandle<float>();

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {

    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            float sum = 0;
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation;
                sdim_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= ssize_t(idim.w)) {
                  continue;
                }
                for (dim_t fd = 0; fd < inCperG; fd++) {
                  sum += float(
                          filterW.at({d, fx, fy, fd}) *
                          inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd}));
                }
              }
            }

            sum += float(biasW.at({d}));
            outW.at({n, ax, ay, d}) = sum;
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}


int convolutionFloat(int8_t* input, int8_t *kernel, int8_t *bias, int8_t *output, dim_t N, dim_t H, dim_t W, dim_t C, dim_t KN, dim_t KH, dim_t KW, int pad_size,
                     int stride_size, size_t group, size_t dilation, bool doRelu, bool doBias, int out_h, int out_w) {
  assert(doRelu==0);
  llvm::SmallVector<dim_t, 4> inputDims = {N,H,W,C};
  Type inputType(ElemKind::FloatTy, inputDims);
  Tensor inV((void*)input, &inputType);

  llvm::SmallVector<dim_t, 4> outputDims = {N,(dim_t)out_h,(dim_t)out_w,KN};
  Type outputType(ElemKind::FloatTy, outputDims);
  Tensor outV((void*)output, &outputType);

  llvm::SmallVector<dim_t, 4> kernelDims = {KN,KH,KW,C};
  Type kernelType(ElemKind::FloatTy, kernelDims);
  Tensor filterV((void*)kernel, &kernelType);

  Tensor* biasV;
  if(doBias) {
    llvm::SmallVector<dim_t, 1> biasDims = {KN};
    Type biasType(ElemKind::FloatTy, biasDims);
    biasV = new Tensor((void *) bias, &biasType);
  }
  else{
    biasV = new Tensor(ElemKind::FloatTy, {KN});
    biasV->zero();
  }

  std::array<unsigned_t,2> kernelS = {(unsigned_t)KH, (unsigned_t)KW};
  llvm::ArrayRef<unsigned_t> kernelSizes(kernelS);

  std::array<unsigned_t,2> strideS = {(unsigned_t)stride_size, (unsigned_t)stride_size};
  llvm::ArrayRef<unsigned_t> strides(strideS);

  std::array<unsigned_t,4> padS = {(unsigned_t)pad_size, (unsigned_t)pad_size, (unsigned_t)pad_size, (unsigned_t)pad_size};
  llvm::ArrayRef<unsigned_t> pads(padS);
  fwdConvolutionInstFloatImpl(
          &inV, &outV, &filterV, biasV,
          kernelSizes, strides,
          pads, group, dilation);
  return 0;
}


void fwdVTAConvolutionInstQuantizedImpl(
        Tensor *inV, Tensor *outV, Tensor *filterV, Tensor *biasV,
        llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
        llvm::ArrayRef<unsigned_t> pads, size_t group, size_t dilation, bool doRelu) {
  auto inW = inV->getHandle<int8_t>();
  auto outW = outV->getHandle<int8_t>();
  auto filterW = filterV->getHandle<int8_t>();
  auto biasW = biasV->getHandle<int32_t>();

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);
  auto outTy = outV->getType();
  auto inTy = inV->getType();
  auto filterTy = filterV->getType();
  auto biasTy = biasV->getType();

  float outScale = outTy.getScale();
  float inScale = inTy.getScale();
  float filterScale = filterTy.getScale();
  float biasScale = biasTy.getScale();
  filterScale = 1/filterScale;
  inScale = 1/inScale;
  biasScale = 1/biasScale;
  outScale = 1/outScale;

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = inScale * filterScale;
  float scale =  matMulScale / outScale;
  float tempScale = 1.0;
  assert(scale > 1);
  uint32_t shift = 0;
  {
    while(tempScale<scale)
    {
      tempScale *= 2;
      shift++;
    }
  }

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            int32_t sum = 0;
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation;
                sdim_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= sdim_t(idim.w)) {
                  continue;
                }
                for (dim_t fd = 0; fd < inCperG; fd++) {

                  int32_t F = filterW.at({d, fx, fy, fd});
                  int32_t I =
                          inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F) * (I);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.

            int32_t B =biasW.at({d});

            // Add the bias.
            sum += B;
            if(doRelu && sum<0) sum = 0;
            // Scale the result back to the expected destination scale.

            outW.at({n, ax, ay, d}) = quantization::clip<int32_t, int8_t>(sum >> shift);

          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}


// (1xK)*(K*8n). width_a = K = depth,  width_b =  * elemsize(1). 
// best performance if K = 16k
// lhs: row-major, rhs: colomn-major
void vector_matrix_multiply_s8(int8_t *lhs_ptr, int8_t *rhs_ptr, int32_t *accum_ptr, int width_a, int width_b, size_t stride_b, int32_t filterOffset)
{
  
  // block = 1x16 * 16*8
  
  //TODO. temp
  // check overflow 
  // upscale to s32 if s16 datatype overflow because of filterOffset  
  assert(filterOffset < 32700 );
  assert(filterOffset > -32700)

  const int16x4_t vfilterOff = vdup_n_s16(filterOffset);

  for (int nblock = 0; nblock < width_b / 8; nblock++)
  {
      // 8 width
      int32x4_t c0[8];

      for (int k = 0; k < 8; k++)
          c0[k] = vdupq_n_s32(0);

      int8_t *a_ptr = lhs_ptr;                         
      int8_t *b_ptr = rhs_ptr + nblock * 8 * stride_b; 
      auto vec_a_end_addr = a_ptr + width_a;

      if( vec_a_end_addr - a_ptr >= 16)
      {
        // This for loop performs 8 accumulations
        // cell layout: 1x16 * 16*8 = 1x8
        for (; a_ptr <= (vec_a_end_addr - 16);)
        {
            const int8x16_t a00_s8 = vld1q_s8(a_ptr);
            const int8x16_t b00_s8 = vld1q_s8(b_ptr + 0 * stride_b);
            const int8x16_t b10_s8 = vld1q_s8(b_ptr + 1 * stride_b);
            const int8x16_t b20_s8 = vld1q_s8(b_ptr + 2 * stride_b);
            const int8x16_t b30_s8 = vld1q_s8(b_ptr + 3 * stride_b);
            const int8x16_t b40_s8 = vld1q_s8(b_ptr + 4 * stride_b);
            const int8x16_t b50_s8 = vld1q_s8(b_ptr + 5 * stride_b);
            const int8x16_t b60_s8 = vld1q_s8(b_ptr + 6 * stride_b);
            const int8x16_t b70_s8 = vld1q_s8(b_ptr + 7 * stride_b);


            // Convert a00_s8 to int16_t and get the lower part
            const int16x4x4_t a00_s16 =
                {
                    {vsub_s16(vget_low_s16(vmovl_s8(vget_low_s8(a00_s8))), vfilterOff),
                    vsub_s16(vget_high_s16(vmovl_s8(vget_low_s8(a00_s8))), vfilterOff),
                    vsub_s16(vget_low_s16(vmovl_s8(vget_high_s8(a00_s8))), vfilterOff),
                    vsub_s16(vget_high_s16(vmovl_s8(vget_high_s8(a00_s8))), vfilterOff)}};

            const int16x4x4_t b00_s16 =
                {
                    {vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))}};

            const int16x4x4_t b10_s16 =
                {
                    {vget_low_s16(vmovl_s8(vget_low_s8(b10_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b10_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b10_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b10_s8)))}};

            const int16x4x4_t b20_s16 =
                {
                    {vget_low_s16(vmovl_s8(vget_low_s8(b20_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b20_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b20_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b20_s8)))}};

            const int16x4x4_t b30_s16 =
                {
                    {vget_low_s16(vmovl_s8(vget_low_s8(b30_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b30_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b30_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b30_s8)))}};

            const int16x4x4_t b40_s16 =
                {
                    {vget_low_s16(vmovl_s8(vget_low_s8(b40_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b40_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b40_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b40_s8)))}};

            const int16x4x4_t b50_s16 =
                {
                    {vget_low_s16(vmovl_s8(vget_low_s8(b50_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b50_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b50_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b50_s8)))}};

            const int16x4x4_t b60_s16 =
                {
                    {vget_low_s16(vmovl_s8(vget_low_s8(b60_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b60_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b60_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b60_s8)))}};

            const int16x4x4_t b70_s16 =
                {
                    {vget_low_s16(vmovl_s8(vget_low_s8(b70_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b70_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b70_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b70_s8)))}};

            // Accumulate 0:
            c0[0] = vmlal_s16(c0[0], b00_s16.val[0], a00_s16.val[0]);
            c0[0] = vmlal_s16(c0[0], b00_s16.val[1], a00_s16.val[1]);
            c0[0] = vmlal_s16(c0[0], b00_s16.val[2], a00_s16.val[2]);
            c0[0] = vmlal_s16(c0[0], b00_s16.val[3], a00_s16.val[3]);

            // Accumulate 1:
            c0[1] = vmlal_s16(c0[1], b10_s16.val[0], a00_s16.val[0]);
            c0[1] = vmlal_s16(c0[1], b10_s16.val[1], a00_s16.val[1]);
            c0[1] = vmlal_s16(c0[1], b10_s16.val[2], a00_s16.val[2]);
            c0[1] = vmlal_s16(c0[1], b10_s16.val[3], a00_s16.val[3]);

            // Accumulate 2:
            c0[2] = vmlal_s16(c0[2], b20_s16.val[0], a00_s16.val[0]);
            c0[2] = vmlal_s16(c0[2], b20_s16.val[1], a00_s16.val[1]);
            c0[2] = vmlal_s16(c0[2], b20_s16.val[2], a00_s16.val[2]);
            c0[2] = vmlal_s16(c0[2], b20_s16.val[3], a00_s16.val[3]);

            // Accumulate 3:
            c0[3] = vmlal_s16(c0[3], b30_s16.val[0], a00_s16.val[0]);
            c0[3] = vmlal_s16(c0[3], b30_s16.val[1], a00_s16.val[1]);
            c0[3] = vmlal_s16(c0[3], b30_s16.val[2], a00_s16.val[2]);
            c0[3] = vmlal_s16(c0[3], b30_s16.val[3], a00_s16.val[3]);

            // Accumulate 4:
            c0[4] = vmlal_s16(c0[4], b40_s16.val[0], a00_s16.val[0]);
            c0[4] = vmlal_s16(c0[4], b40_s16.val[1], a00_s16.val[1]);
            c0[4] = vmlal_s16(c0[4], b40_s16.val[2], a00_s16.val[2]);
            c0[4] = vmlal_s16(c0[4], b40_s16.val[3], a00_s16.val[3]);

            // Accumulate 5:
            c0[5] = vmlal_s16(c0[5], b50_s16.val[0], a00_s16.val[0]);
            c0[5] = vmlal_s16(c0[5], b50_s16.val[1], a00_s16.val[1]);
            c0[5] = vmlal_s16(c0[5], b50_s16.val[2], a00_s16.val[2]);
            c0[5] = vmlal_s16(c0[5], b50_s16.val[3], a00_s16.val[3]);

            // Accumulate 6:
            c0[6] = vmlal_s16(c0[6], b60_s16.val[0], a00_s16.val[0]);
            c0[6] = vmlal_s16(c0[6], b60_s16.val[1], a00_s16.val[1]);
            c0[6] = vmlal_s16(c0[6], b60_s16.val[2], a00_s16.val[2]);
            c0[6] = vmlal_s16(c0[6], b60_s16.val[3], a00_s16.val[3]);

            // Accumulate 7:
            c0[7] = vmlal_s16(c0[7], b70_s16.val[0], a00_s16.val[0]);
            c0[7] = vmlal_s16(c0[7], b70_s16.val[1], a00_s16.val[1]);
            c0[7] = vmlal_s16(c0[7], b70_s16.val[2], a00_s16.val[2]);
            c0[7] = vmlal_s16(c0[7], b70_s16.val[3], a00_s16.val[3]);

            a_ptr += 16;
            b_ptr += 16;

#if DEBUG_INFO
  if(nblock==0) {
    
    int32_t sum_temp[4];
    vst1q_s32(sum_temp, c0[0]);

    printf("step:%d - ", a_ptr - lhs_ptr);
    for(int k=0;k<4;k++)
    { 
      printf(" %d ", sum_temp[k]); 
    }
    printf("\n");

  }
#endif    

        }
      }

      // This for loop performs the left-over accumulations
      // left elem is less than 16
      {
          
          const int8x16_t a00_s8 = vld1q_s8(a_ptr);
          const int8x16_t b00_s8 = vld1q_s8(b_ptr + 0 * stride_b);
          const int8x16_t b10_s8 = vld1q_s8(b_ptr + 1 * stride_b);
          const int8x16_t b20_s8 = vld1q_s8(b_ptr + 2 * stride_b);
          const int8x16_t b30_s8 = vld1q_s8(b_ptr + 3 * stride_b);
          const int8x16_t b40_s8 = vld1q_s8(b_ptr + 4 * stride_b);
          const int8x16_t b50_s8 = vld1q_s8(b_ptr + 5 * stride_b);
          const int8x16_t b60_s8 = vld1q_s8(b_ptr + 6 * stride_b);
          const int8x16_t b70_s8 = vld1q_s8(b_ptr + 7 * stride_b);

          // Convert a00_s8 to int16_t and get the lower part
          const int16x4x4_t a00_s16 =
              {
                {vsub_s16(vget_low_s16(vmovl_s8(vget_low_s8(a00_s8))), vfilterOff),
                    vsub_s16(vget_high_s16(vmovl_s8(vget_low_s8(a00_s8))), vfilterOff),
                    vsub_s16(vget_low_s16(vmovl_s8(vget_high_s8(a00_s8))), vfilterOff),
                    vsub_s16(vget_high_s16(vmovl_s8(vget_high_s8(a00_s8))), vfilterOff)}};

          const int16x4x4_t b00_s16 =
              {
                  {vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))}};

          const int16x4x4_t b10_s16 =
              {
                  {vget_low_s16(vmovl_s8(vget_low_s8(b10_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b10_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b10_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b10_s8)))}};

          const int16x4x4_t b20_s16 =
              {
                  {vget_low_s16(vmovl_s8(vget_low_s8(b20_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b20_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b20_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b20_s8)))}};

          const int16x4x4_t b30_s16 =
              {
                  {vget_low_s16(vmovl_s8(vget_low_s8(b30_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b30_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b30_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b30_s8)))}};

          const int16x4x4_t b40_s16 =
              {
                  {vget_low_s16(vmovl_s8(vget_low_s8(b40_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b40_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b40_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b40_s8)))}};

          const int16x4x4_t b50_s16 =
              {
                  {vget_low_s16(vmovl_s8(vget_low_s8(b50_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b50_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b50_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b50_s8)))}};

          const int16x4x4_t b60_s16 =
              {
                  {vget_low_s16(vmovl_s8(vget_low_s8(b60_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b60_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b60_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b60_s8)))}};

          const int16x4x4_t b70_s16 =
              {
                  {vget_low_s16(vmovl_s8(vget_low_s8(b70_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b70_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b70_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b70_s8)))}};

            int ii = 0; // index 
            if(vec_a_end_addr - a_ptr  >= 8) {
            // Accumulate 0:

              c0[0] = vmlal_s16(c0[0], b00_s16.val[0+ii], a00_s16.val[0+ii]);
              c0[0] = vmlal_s16(c0[0], b00_s16.val[1+ii], a00_s16.val[1+ii]);

              // Accumulate 1:
              c0[1] = vmlal_s16(c0[1], b10_s16.val[0+ii], a00_s16.val[0+ii]);
              c0[1] = vmlal_s16(c0[1], b10_s16.val[1+ii], a00_s16.val[1+ii]);

              // Accumulate 2:
              c0[2] = vmlal_s16(c0[2], b20_s16.val[0+ii], a00_s16.val[0+ii]);
              c0[2] = vmlal_s16(c0[2], b20_s16.val[1+ii], a00_s16.val[1+ii]);

              // Accumulate 3:
              c0[3] = vmlal_s16(c0[3], b30_s16.val[0+ii], a00_s16.val[0+ii]);
              c0[3] = vmlal_s16(c0[3], b30_s16.val[1+ii], a00_s16.val[1+ii]);

              // Accumulate 4:
              c0[4] = vmlal_s16(c0[4], b40_s16.val[0+ii], a00_s16.val[0+ii]);
              c0[4] = vmlal_s16(c0[4], b40_s16.val[1+ii], a00_s16.val[1+ii]);

              // Accumulate 5:
              c0[5] = vmlal_s16(c0[5], b50_s16.val[0+ii], a00_s16.val[0+ii]);
              c0[5] = vmlal_s16(c0[5], b50_s16.val[1+ii], a00_s16.val[1+ii]);

              // Accumulate 6:
              c0[6] = vmlal_s16(c0[6], b60_s16.val[0+ii], a00_s16.val[0+ii]);
              c0[6] = vmlal_s16(c0[6], b60_s16.val[1+ii], a00_s16.val[1+ii]);

              // Accumulate 7:
              c0[7] = vmlal_s16(c0[7], b70_s16.val[0+ii], a00_s16.val[0+ii]);
              c0[7] = vmlal_s16(c0[7], b70_s16.val[1+ii], a00_s16.val[1+ii]);

              a_ptr+=8;
              ii+= (8/4);
            }

            if(vec_a_end_addr - a_ptr >= 4) {
              // Accumulate 0:
              c0[0] = vmlal_s16(c0[0], b00_s16.val[0+ii], a00_s16.val[0+ii]);

              // Accumulate 1:
              c0[1] = vmlal_s16(c0[1], b10_s16.val[0+ii], a00_s16.val[0+ii]);

              // Accumulate 2:
              c0[2] = vmlal_s16(c0[2], b20_s16.val[0+ii], a00_s16.val[0+ii]);

              // Accumulate 3:
              c0[3] = vmlal_s16(c0[3], b30_s16.val[0+ii], a00_s16.val[0+ii]);

              // Accumulate 4:
              c0[4] = vmlal_s16(c0[4], b40_s16.val[0+ii], a00_s16.val[0+ii]);

              // Accumulate 5:
              c0[5] = vmlal_s16(c0[5], b50_s16.val[0+ii], a00_s16.val[0+ii]);

              // Accumulate 6:
              c0[6] = vmlal_s16(c0[6], b60_s16.val[0+ii], a00_s16.val[0+ii]);

              // Accumulate 7:
              c0[7] = vmlal_s16(c0[7], b70_s16.val[0+ii], a00_s16.val[0+ii]);

              a_ptr+=4;
              ii+= (4/4);
            }

          int left = vec_a_end_addr - a_ptr;
          if(left > 0 )
          {
            int16_t l1_mask[4] = { 1 , 0 ,0 ,0};
            int16_t l2_mask[4] = { 1 , 1 ,0 ,0};
            int16_t l3_mask[4] = { 1 , 1 ,1 ,0};
            int16x4_t left_mask;
            
            if(left==1) left_mask = vld1_s16(l1_mask);
            else if(left==2) left_mask = vld1_s16(l2_mask);
            else left_mask = vld1_s16(l3_mask);

            // Accumulate 0:
            c0[0] = vmlal_s16(c0[0], b00_s16.val[0+ii], vmul_s16(left_mask, a00_s16.val[0+ii]));

            // Accumulate 1:
            c0[1] = vmlal_s16(c0[1], b10_s16.val[0+ii], vmul_s16(left_mask, a00_s16.val[0+ii]));

            // Accumulate 2:
            c0[2] = vmlal_s16(c0[2], b20_s16.val[0+ii], vmul_s16(left_mask, a00_s16.val[0+ii]));

            // Accumulate 3:
            c0[3] = vmlal_s16(c0[3], b30_s16.val[0+ii], vmul_s16(left_mask, a00_s16.val[0+ii]));

            // Accumulate 4:
            c0[4] = vmlal_s16(c0[4], b40_s16.val[0+ii], vmul_s16(left_mask, a00_s16.val[0+ii]));

            // Accumulate 5:
            c0[5] = vmlal_s16(c0[5], b50_s16.val[0+ii], vmul_s16(left_mask, a00_s16.val[0+ii]));

            // Accumulate 6:
            c0[6] = vmlal_s16(c0[6], b60_s16.val[0+ii], vmul_s16(left_mask, a00_s16.val[0+ii]));

            // Accumulate 7:
            c0[7] = vmlal_s16(c0[7], b70_s16.val[0+ii], vmul_s16(left_mask, a00_s16.val[0+ii]));


            a_ptr += left;
          }
          // b_ptr += stride_b;
      }


      for (int k = 0; k < 8; k++)
      {
          int32x2_t temp = vadd_s32(vget_high_s32(c0[k]), vget_low_s32(c0[k]));
          int32x2_t temp2 = vpadd_s32(temp, temp);
          int32_t sum1 = vget_lane_s32(temp2, 0);

          //vst1q_s32(accum_ptr + nblock * 8 + k, c0[k]);
          accum_ptr[nblock * 8  + k] = sum1;
    
      }
  }

   
}

// add bias offset for output, scaling and clip from s32 to s8.
// out = (int8)((input * matscale) + bias * vscale)
// ! bias is type int32_t. bias are same for a channel(row)
// width = 8n
void vector_scale_bias_offset_clip_s8(int32_t *input, int32_t biasValue, int8_t *output, float32_t scale, float32_t bscale, int width, bool doBias, bool doRelu,int32_t outOff)
{
    
  const float32x4_t scale_v = vdupq_n_f32(scale);
  const float32x4_t bscale_v = vdupq_n_f32(bscale);
  const int32x4_t zero_v = vdupq_n_s32(0);

  // clip
  const int32x4_t max = vdupq_n_s32(127);
  const int32x4_t min = vdupq_n_s32(-128);

  //out offset
  const int32x4_t voff = vdupq_n_s32(outOff);

#if DEBUG_INFO    
printf("doBias:%d doRelu:%d\n",doBias,doRelu);
printf("s:%f bs:%f -- ",scale, bscale);
printf("o:%p - %p\n",output, output+width);  
#endif  

  if(!doBias) {
    for (int i=0; i<width;i+=8)
    {
        const int32x4_t a0_s32 = vld1q_s32(input+i);
        const int32x4_t a1_s32 = vld1q_s32(input+i+4);
        
#if DEBUG_INFO  
        int32_t aa[4];
        if(i==0) {
          vst1q_s32(aa,b00);
          printf("%d %d %d %d ---\n",aa[0],aa[1],aa[2],aa[3]);
          vst1q_s32(aa,b01);
          printf("%d %d %d %d ---\n",aa[0],aa[1],aa[2],aa[3]);
          
        }
#endif
        int16x4_t vres1, vres2;
        if(doRelu)
        {
          vres1   = vmovn_s32(vmaxq_s32(vminq_s32(vaddq_s32(voff,vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmaxq_s32(zero_v,a0_s32)),scale_v))), max),min));
          vres2   = vmovn_s32(vmaxq_s32(vminq_s32(vaddq_s32(voff,vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmaxq_s32(zero_v,a1_s32)),scale_v))), max),min));
          
        }
        else {
          vres1   = vmovn_s32(vmaxq_s32(vminq_s32(vaddq_s32(voff,vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(a0_s32),scale_v))), max),min));
          vres2   = vmovn_s32(vmaxq_s32(vminq_s32(vaddq_s32(voff,vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(a1_s32),scale_v))), max),min));

        }

        int8x8_t vres = vmovn_s16(vcombine_s16(vres1,vres2));
        vst1_s8(output+i, vres);
        
    }      
  }
  else if(doBias && bscale == 1.0) {
      const int32x4_t v_bias = vdupq_n_s32(biasValue);

    for (int i=0; i<width;i+=8)
    {
        const int32x4_t a0_s32 = vld1q_s32(input+i);
        const int32x4_t a1_s32 = vld1q_s32(input+i+4);

        
#if DEBUG_INFO  
        int32_t aa[4];
        if(i==0) {
          vst1q_s32(aa,a0_s32);
          printf("%d %d %d %d ---\n",aa[0],aa[1],aa[2],aa[3]);
          vst1q_s32(aa,a1_s32);
          printf("%d %d %d %d ---\n",aa[0],aa[1],aa[2],aa[3]);
          
        }
#endif


        int16x4_t vres1, vres2;
        
        if(doRelu)
        {
          vres1   = vmovn_s32(vmaxq_s32(vminq_s32(vaddq_s32(voff,vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmaxq_s32(zero_v,vaddq_s32(v_bias,a0_s32))),scale_v))), max),min));
          vres2   = vmovn_s32(vmaxq_s32(vminq_s32(vaddq_s32(voff,vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmaxq_s32(zero_v,vaddq_s32(v_bias,a1_s32))),scale_v))), max),min));

        }
        else {
          vres1   = vmovn_s32(vmaxq_s32(vminq_s32(vaddq_s32(voff,vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(v_bias,a0_s32)),scale_v))), max),min));
          vres2   = vmovn_s32(vmaxq_s32(vminq_s32(vaddq_s32(voff,vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(v_bias,a1_s32)),scale_v))), max),min));

        }
#if DEBUG_INFO        
        int16_t dd[4];
        if(i==0) {
          vst1_s16(dd,vres1);
          printf("%d %d %d %d ---\n",dd[0],dd[1],dd[2],dd[3]);
          vst1_s16(dd,vres2);
          printf("%d %d %d %d ---\n",dd[0],dd[1],dd[2],dd[3]);
          
        }        
#endif  
        int8x8_t vres = vmovn_s16(vcombine_s16(vres1,vres2));

        vst1_s8(output+i, vres);

    }      
  }
  else {
    printf("[ERR] bscale = %f\n. not supported bias scale\n", bscale);
    return;
  }

}

void fwdConvolutionInstQuantizedImpl(
        Tensor *inV, Tensor *outV, Tensor *filterV, Tensor *biasV,
        llvm::ArrayRef<unsigned_t> kernelSizes, llvm::ArrayRef<unsigned_t> strides,
        llvm::ArrayRef<unsigned_t> pads, size_t group, size_t dilation, bool doRelu) {
  auto inW = inV->getHandle<int8_t>();
  auto outW = outV->getHandle<int8_t>();
  auto filterW = filterV->getHandle<int8_t>();
  auto biasW = biasV->getHandle<int32_t>();

  ShapeNHWC odim(outW.dims());
  ShapeNHWC idim(inW.dims());
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  assert(idim.c % group == 0 && "Input channels must be divisible by group.");
  assert(odim.c % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = idim.c / group;
  dim_t outCperG = odim.c / group;

  PaddingTLBR pdim(pads);
  auto outTy = outV->getType();
  auto inTy = inV->getType();
  auto filterTy = filterV->getType();
  auto biasTy = biasV->getType();

  int32_t outOffset = outTy.getOffset();
  int32_t inOffset = inTy.getOffset();
  int32_t filterOffset = filterTy.getOffset();
  int32_t biasOffset = biasTy.getOffset();

  float outScale = outTy.getScale();
  float inScale = inTy.getScale();
  float filterScale = filterTy.getScale();
  float biasScale = biasTy.getScale();

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = inScale * filterScale;

  // For each input in the batch:
  for (dim_t n = 0; n < idim.n; n++) {
    // For each group of input channels:
    for (dim_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pdim.top);
        for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
          ssize_t y = -ssize_t(pdim.left);
          for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

            // For each element in the convolution-filter:
            int32_t sum = 0;
            for (dim_t fx = 0; fx < kdim.height; fx++) {
              for (dim_t fy = 0; fy < kdim.width; fy++) {
                sdim_t ox = x + fx * dilation;
                sdim_t oy = y + fy * dilation;

                // Ignore index access below zero (this is due to padding).
                if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                    oy >= sdim_t(idim.w)) {
                  continue;
                }
                for (dim_t fd = 0; fd < inCperG; fd++) {

                  int32_t F = filterW.at({d, fx, fy, fd});
                  int32_t I =
                          inW.at({n, (dim_t)ox, (dim_t)oy, g * inCperG + fd});
                  // We represent the element multiplication with offset as
                  // (value - offset).
                  sum += (F - filterOffset) * (I - inOffset);
                }
              }
            }

            // Scale the bias to match the scale of the matrix multiplication.

            int32_t B =std::round(float(biasW.at({d}) - biasOffset) *
                                  (biasScale / matMulScale));

            // Add the bias.
            sum += B;

            if(doRelu && sum<0){
              sum = 0;
            }
            // Scale the result back to the expected destination scale.

            outW.at({n, ax, ay, d}) = quantization::clip<int32_t, int8_t>(
                    std::round(float(sum) * (matMulScale / outScale) + outOffset));
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

int nonvtaconvolution(int8_t* input, float inputScale, int32_t inputOffset, int8_t *kernel, float kernelScale, int32_t kernelOffset,
                      int8_t *bias, float biasScale, int32_t biasOffset, int8_t *output, float outputScale, int32_t outputOffset, dim_t N, dim_t H, dim_t W, dim_t C, dim_t KN, dim_t KH, dim_t KW, int pad_size,
                      int stride_size, size_t group, size_t dilation, bool doRelu, bool doBias, int out_h, int out_w) {
  llvm::SmallVector<dim_t, 4> inputDims = {N,H,W,C};
  Type inputType(ElemKind::Int8QTy, inputDims, inputScale, inputOffset);
  Tensor inV((void*)input, &inputType);

  llvm::SmallVector<dim_t, 4> outputDims = {N,(dim_t)out_h,(dim_t)out_w,KN};
  Type outputType(ElemKind::Int8QTy, outputDims, outputScale, outputOffset);
  Tensor outV((void*)output, &outputType);

  llvm::SmallVector<dim_t, 4> kernelDims = {KN,KH,KW,C};
  Type kernelType(ElemKind::Int8QTy, kernelDims, kernelScale, kernelOffset);
  Tensor filterV((void*)kernel, &kernelType);

  Tensor* biasV;
  if(doBias) {
    llvm::SmallVector<dim_t, 1> biasDims = {KN};
    Type biasType(ElemKind::Int32QTy, biasDims, biasScale, biasOffset);
    biasV = new Tensor((void *) bias, &biasType);
  }
  else{
    biasV = new Tensor(ElemKind::Int32QTy, {KN}, 1.0, 0);
    biasV->zero();
  }

  std::array<unsigned_t,2> kernelS = {(unsigned_t)KH, (unsigned_t)KW};
  llvm::ArrayRef<unsigned_t> kernelSizes(kernelS);

  std::array<unsigned_t,2> strideS = {(unsigned_t)stride_size, (unsigned_t)stride_size};
  llvm::ArrayRef<unsigned_t> strides(strideS);

  std::array<unsigned_t,4> padS = {(unsigned_t)pad_size, (unsigned_t)pad_size, (unsigned_t)pad_size, (unsigned_t)pad_size};
  llvm::ArrayRef<unsigned_t> pads(padS);






  float filterScale = kernelScale;
  filterScale = 1/filterScale;
  float inScale = inputScale;
  inScale = 1/inScale;
  //float biasScale = biasScale;
  biasScale = 1/biasScale;
  float matMulScale = inScale * filterScale;
  float outScale = outputScale;
  outScale = 1/outScale;
  float scale =  matMulScale / outScale;
  float tempScale = 1.0;
  assert(scale > 1);
  uint32_t shift = 0;
  {
    while(tempScale<scale)
    {
      tempScale *= 2;
      shift++;
    }
  }

  int32_t outOffset = outputOffset;
  int32_t inOffset = inputOffset;
  int32_t filterOffset = kernelOffset;
  //int32_t biasOffset = biasOffset;
  if(C%16 ==0 && KN%16 ==0 && tempScale==scale && biasScale==matMulScale && outOffset ==0 && inOffset ==0 && filterOffset ==0 && biasOffset ==0) {
    fwdVTAConvolutionInstQuantizedImpl(
            &inV, &outV, &filterV, biasV,
            kernelSizes, strides,
            pads, group, dilation, doRelu);
  }
  else {
    fwdConvolutionInstQuantizedImpl(
            &inV, &outV, &filterV, biasV,
            kernelSizes, strides,
            pads, group, dilation, doRelu);
  }
  return 0;
}


template <typename ElemTy, typename AccumulatorTy, typename BiasElemTy>
void fwdFullyConnectedInstQuantizedImpl(Tensor *inV, Tensor *outV, Tensor *weightsV, Tensor *biasV){
  //const glow::FullyConnectedInst *I) {

  auto inW = inV->getHandle<ElemTy>();
  auto outW = outV->getHandle<ElemTy>();
  auto weightsW = weightsV->getHandle<ElemTy>();
  auto biasW = biasV->getHandle<BiasElemTy>();

  auto inTy = inW.getType();
  auto weightsTy = weightsW.getType();
  auto biasTy = biasW.getType();
  auto outTy = outW.getType();

  int32_t inOffset = inTy.getOffset();
  int32_t weightsOffset = weightsTy.getOffset();
  int32_t biasOffset = biasTy.getOffset();
  int32_t outOffset = outTy.getOffset();

  float outScale = outTy.getScale();
  float weightsScale = weightsTy.getScale();
  float biasScale = biasTy.getScale();
  float inScale = inTy.getScale();

  ShapeHW idim(inW.dims());
  ShapeHW odim(outW.dims());

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
  float matMulScale = weightsScale * inScale;

  outW.clear(0);

  for (dim_t i = 0; i < idim.height; i++) {
    for (dim_t j = 0; j < odim.width; j++) {
      AccumulatorTy sum = 0;
      for (dim_t k = 0; k < idim.width; k++) {
        AccumulatorTy W = weightsW.at({k, j});
        AccumulatorTy A = inW.at({i, k});
        sum += (W - weightsOffset) * (A - inOffset);
      }

      // Scale the bias to match the scale of the matrix multiplication.
      AccumulatorTy B = std::round(float(biasW.at({j}) - biasOffset) *
                                   (biasScale / matMulScale));

      // Add the bias.
      sum += B;

      // Scale the result back to the expected destination scale.
      outW.at({i, j}) = quantization::clip<AccumulatorTy, ElemTy>(
              std::round(float(sum) * (matMulScale / outScale)) + outOffset);
    }
  }
}

int fullyconnected(int8_t* input, float inputScale, int32_t inputOffset, int8_t *kernel, float kernelScale, int32_t kernelOffset,
                   int8_t *bias, float biasScale, int32_t biasOffset, int8_t *output, float outputScale, int32_t outputOffset,
                   dim_t inputDim0, dim_t inputDim1, dim_t kernelDim0, dim_t kernelDim1, dim_t outputDim0, dim_t outputDim1, bool doBias) {
#if DEBUG_INFO		  
   printf("fullly: in(%d.%d) is:%f io:%d -- kernel(%d,%d) ks:%f ko:%d -- out(%d,%d) os:%f oo:%d , doBais:%d bs:%f bo:%d\n",
   inputDim0, inputDim1, inputScale, inputOffset, kernelDim0, kernelDim1, kernelScale, kernelOffset, outputDim0, outputDim1, outputScale, outputOffset, doBias,biasScale,biasOffset);
#endif 
    
 if(inputDim1 != kernelDim0 || inputDim1%16 !=0 ||  kernelDim1%4 !=0) {
   printf("[ERROR] invalid layout\n");
   return -1;
 }

 // layout: input = row-major, kernel => row-major. convrt row-major to column-major
  if(inputDim0 == 1 )
  {
      int8_t* K = (int8_t*)malloc( inputDim1 * kernelDim1);
      int32_t* C = (int32_t*)malloc(kernelDim1);
      
      //TODO neon
      // conv row to  column-major
      for(int i=0;i<kernelDim1;i++)
        for(int j=0;j<inputDim1;j++)
          *(K+ inputDim1*i + j) = kernel[i+j*kernelDim1];

      // K=col-major
      float matMulScale = kernelScale * inputScale;
      float scale = matMulScale / outputScale;
      float bscale = biasScale / matMulScale;
 
      vector_matrix_multiply_s8(input, K, (int32_t *)C, inputDim1, kernelDim1, kernelDim0,kernelOffset );
      vector_scale_bias_offset_clip_s8(C, *((int32_t*)bias), output, scale,bscale, kernelDim1, 1, 0,outputOffset);
#if DEBUG_INFO
      for(int k=0;k<kernelDim1;k++) {
        int32_t *B = (int32_t*)bias;
        printf("[%d]i32:%d (%d+%d) output[k]=%d  ",k, C[k]+B[k],C[k],B[k],output[k]);
        if(k%4==0) printf("\n");
      }
       printf("\n\n (0,0) b0:%d C0:%d out0:%d",bias[0], C[0],output[0]);
#endif       

  }

  return 0;
}



void maxpool_nxn_nhwc(int8_t *input, int8_t *output, dim_t N, dim_t H, dim_t W, dim_t C,
            unsigned_t KH, unsigned_t KW, unsigned_t pad_size,
            unsigned_t stride_size) {
  
  //TODO padding size
  
  // kernel size = nXn
  // nhwc. channel should be multiple of 16 to use int8x16_t type.
  if(KH!=KW && pad_size!=1) {
    printf("!!!!!!!!!! not impl yet\n");
    return;
  }

  int p=(int)pad_size;
  int out_w = ((W+2*p + 1) - (KW - 1))/stride_size;
  int out_h = ((H+2*p +1) - (KH - 1))/stride_size;

#if DEBUG_INFO  
  printf("out_w =%d, out_h=%d\n",out_w, out_h);
#endif  

  for(int n=0;n<N;n++)
  {
    for(int c=0;c<C/16;c++) // 16 channel once
    {
      for(int h=0,ho=0;h<H+2*p;h+=stride_size,ho++)
      {
        for(int w=0,wo=0;w<W+2*p;w+=stride_size,wo++)
        {

          // assume kernel 3x3
          // pass if padding. start 0,0 including padding
          const int pool_start_y = std::max(p, h);
          const int pool_end_y   = std::min((int)H+p, (int)(h+KH));
          const int pool_start_x = std::max(p, w);
          const int pool_end_x   = std::min((int)W+p, (int)(w+KW));
          
          int8x16_t vres = vdupq_n_s8(-128);  // min value
#if DEBUG_INFO
          printf("--[%d~%d, %d~%d]--\n",pool_start_x,pool_end_x,pool_start_y,pool_end_y);
#endif
          // pool start index
          const int nopad_x=pool_start_x-p;
          const int nopad_y=pool_start_y-p;
          int start_idx = n*H*W*C + nopad_y*W*C + nopad_x*C + (c*16); 

          for(int y = 0; y < pool_end_y - pool_start_y; ++y)
          {
              for(int x = 0; x < pool_end_x - pool_start_x; ++x)
              {
                //load a single vector
                const int8x16_t data = vld1q_s8(input + start_idx + x*C + y*W*C);
#if DEBUG_INFO
                //debug
                int8_t at[16];
                vst1q_s8(at, data);
                printf("[%d,%d]%x(%p) ",x,y,at[0],input + start_idx + x*C + y*W*C);
#endif
                // Vr[i] := (Va[i] >= Vb[i]) ? Va[i] : Vb[i]
                vres = vmaxq_s8(vres, data);
              }
          }

          // Store result
          // TODO check if qinfo matters. in orignal code, true means input , output scale offsets are same
          // org code -> (input_qinfo != output_qinfo) ? vrequantize_pooling<q8x8_t, q8x16_t>(wrapper::vgetlow(vres), wrapper::vgethigh(vres), requant_qinfo) :
          int out_idx = n* out_w*out_w*C +ho*out_w*C + wo*C + (c*16);
          vst1q_s8(output+out_idx,  vres);
#if DEBUG_INFO
          //debug
          int8_t ot[16];
          vst1q_s8(ot, vres);
          printf("\n out = %x\n ",ot[0]);
#endif
          
        }
      }
    }
  }
#if DEBUG_INFO
  for(int j=0;j<out_w*out_h;j++)
  {
    if(j%16==0) printf("\n");
    printf("%x ", output[j]);
  }
#endif  

}



int maxpool(int8_t* input, int8_t *output, dim_t N, dim_t H, dim_t W, dim_t C, unsigned_t KH, unsigned_t KW, unsigned_t pad_size,
            unsigned_t stride_size){

  printf("maxpool kh=%d kw=%d pad=%d str=%d DIM=N:%d H:%d W:%d C:%d\n", KH,KW,pad_size,stride_size, N,H,W,C );

  maxpool_nxn_nhwc(input,output,N,H,W,C,KH,KW,pad_size,stride_size);
  return 0;

}

int relu(int8_t* input, int8_t *output, unsigned_t size, float inputScale, float outputScale){

  int int_scale = (int)(inputScale / outputScale);
  const int8x16_t scale_v = vdupq_n_s8(int_scale);
  const int8x16_t zero_v = vdupq_n_s8(0);
  int div16_cnt = size / 16;
  int offset=0;
  for (int i = 0; i < div16_cnt ; i++) {
    const int8x16_t a = vld1q_s8(input+i*16);
    if(int_scale != 1) vst1q_s8(output+i*16, vmulq_s8(vmaxq_s8(zero_v,a),scale_v));
    else vst1q_s8(output+i*16, vmaxq_s8(zero_v,a));
  }
  if(size%16 >=8)
  {
    const int8x8_t a8 = vld1_s8(input+div16_cnt*16);
    const int8x8_t scale_v8 = vdup_n_s8(int_scale);
    const int8x8_t zero_v8 = vdup_n_s8(0);
    if(int_scale != 1) vst1_s8(output+div16_cnt*16, vmul_s8(vmax_s8(zero_v8,a8),scale_v8));
    else vst1_s8(output+div16_cnt*16, vmax_s8(zero_v8,a8));
    offset+=8;
  }
  int left = size - div16_cnt*16 - offset;
  for (int i = 0; i < left; i++) {
    int idx=div16_cnt*16 + offset + i;
    if (input[idx] > 0) output[idx] = input[idx];
    else output[idx] = 0;
    output[idx] *= int_scale;
  }

  return 0;
}

void fwdAvgPoolInstI8Impl(Tensor *inW, Tensor *outW,
                          llvm::ArrayRef<unsigned_t> kernelSizes,
                          llvm::ArrayRef<unsigned_t> strides,
                          llvm::ArrayRef<unsigned_t> pads) {
  ShapeNHWC odim(outW->dims());
  ShapeNHWC idim(inW->dims());

  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  Handle<int8_t> inHandle = inW->getHandle<int8_t>();
  Handle<int8_t> outHandle = outW->getHandle<int8_t>();

  // Implement the avg pooling operation as defined here:
  // https://arxiv.org/abs/1312.4400
  float filterArea = kdim.height * kdim.width;

  TensorQuantizationParams inQP{inW->getType().getScale(),
                                inW->getType().getOffset()};
  TensorQuantizationParams outQP{outW->getType().getScale(),
                                 outW->getType().getOffset()};

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {
    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pdim.top);
      for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        ssize_t y = -ssize_t(pdim.left);
        for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {
          int32_t sum = 0;

          for (dim_t fx = 0; fx < kdim.height; fx++) {
            for (dim_t fy = 0; fy < kdim.width; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              sum += inHandle.at({n, (dim_t)ox, (dim_t)oy, z}) - inQP.offset;
            }
          }
          // Instead of dividing by filterArea, just change scale.
          outHandle.at({n, ax, ay, z}) = quantization::clip<int32_t, int8_t>(
                  std::round(float(sum) * (inQP.scale / outQP.scale / filterArea) +
                             outQP.offset));
        } // W
      }   // H
    }     // C
  }       // N

  return;
}


void avgpool_nxn_nhwc(int8_t *input, float inputScale, int32_t inputOffset, int8_t *output, float outputScale, int32_t outputOffset, dim_t N, dim_t H, dim_t W, dim_t C,
           dim_t outputDim0, dim_t outputDim1, dim_t outputDim2, dim_t outputDim3,unsigned_t KH, unsigned_t KW, unsigned_t pad_size,
            unsigned_t stride_size) {
  
  //TODO padding > 1

  // kernel size = nXn
  // nhwc. channel should be multiple of 16
  if(KH!=KW && pad_size!=1) {
    printf("!!!!!!!!!! not impl yet\n");
    return;
  }

  const float32x4_t             half_scale_v = vdupq_n_f32(0.5f);
  int p=(int)pad_size;

  const float scale = 1.0f * (inputScale / outputScale / (KW*KH)); 

#if DEBUG_INFO
  printf("scale:%f\n",scale);
#endif  

  for(int n=0;n<N;n++)
  {
    for(int c=0;c<C/16;c++) 
    {
        for(int h=0,ho=0;h<H+2*p-KH+1;h+=stride_size,ho++) // ho = h out.
        {
          for(int w=0,wo=0;w<W+2*p-KW+1;w+=stride_size,wo++)
          {


          const int pool_start_y = std::max(p, h);
          const int pool_end_y   = std::min((int)H+p, (int)(h+KH));
          const int pool_start_x = std::max(p, w);
          const int pool_end_x   = std::min((int)W+p, (int)(w+KW));
          
  
          float32x4_t vres1 = vdupq_n_f32(0.f);
          float32x4_t vres2 = vdupq_n_f32(0.f);
          float32x4_t vres3 = vdupq_n_f32(0.f);
          float32x4_t vres4 = vdupq_n_f32(0.f);



#if DEBUG_INFO
          printf("--[%d~%d, %d~%d]--\n",pool_start_x,pool_end_x,pool_start_y,pool_end_y);
#endif
          // pool start idx
          const int nopad_x=pool_start_x-p;
          const int nopad_y=pool_start_y-p;
          int start_idx = n*H*W*C + nopad_y*W*C + nopad_x*C + (c*16); 


          for(int y = 0; y < pool_end_y - pool_start_y; ++y)
          {
              for(int x = 0; x < pool_end_x - pool_start_x; ++x)
              {
                //load a single vector
                const int8x16_t data = vld1q_s8(input + start_idx + x*C + y*W*C);

                const int16x8_t data_s16  = vmovl_s8(vget_low_s8(data));
                const int16x8_t data2_s16 = vmovl_s8(vget_high_s8(data));

                vres1   = vaddq_f32(vres1, vcvtq_f32_s32(vmovl_s16(vget_low_s16(data_s16))));
                vres2   = vaddq_f32(vres2, vcvtq_f32_s32(vmovl_s16(vget_high_s16(data_s16))));
                vres3   = vaddq_f32(vres3, vcvtq_f32_s32(vmovl_s16(vget_low_s16(data2_s16))));
                vres4   = vaddq_f32(vres4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(data2_s16))));
    
#if DEBUG_INFO
                //debug
                int8_t at[16];
                float32_t tt[4];
                vst1q_f32(tt,vres1);
                vst1q_s8(at, data);
                printf("[%d,%d]%x(%f)  ",x,y,at[0],vres1[0]); // %p  ,input + start_idx + x*C + y*W*C);
#endif

              }
          }

          //TODO check using vcvtaq or vcvtnq
          // Store result
          const float32x4_t scale_v = vdupq_n_f32(scale);
          // Divide by scale and add 0.5f to round to nearest instead of rounding towards zero
          vres1 = vmlaq_f32(half_scale_v, vres1, scale_v);
          vres2 = vmlaq_f32(half_scale_v, vres2, scale_v);
          vres3 = vmlaq_f32(half_scale_v, vres3, scale_v);
          vres4 = vmlaq_f32(half_scale_v, vres4, scale_v);

          // vcvtq round towrd zero shows same result with original one
          const int8x8_t res1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vres1)), vqmovn_s32(vcvtq_s32_f32(vres2))));
          const int8x8_t res2 = vqmovn_s16(vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vres3)), vqmovn_s32(vcvtq_s32_f32(vres4))));
          int out_idx = n*((H+2*p)/stride_size)*((W+2*p)/stride_size)*C +ho*((W+2*p)/stride_size)*C + wo*C + (c*16);
          const int8x16_t vres = vcombine_s8(res1, res2);
          vst1q_s8(output + out_idx, vres);

 
#if DEBUG_INFO
          //debug
          float32_t t1[4];
          vst1q_f32(t1, vres1);
          int8_t ot[16];
          vst1q_s8(ot, vres);
          printf("\n out = %f %d [%d]\n ",t1[0],ot[0],out_compare[out_idx]);
#endif
          
        }
      }
    }
  }
#if DEBUG_INFO
  for(int j=0;j<1024;j++)
  {
    if(j%16==0) printf("\n");
    printf("%x ", output[j]);
  }
#endif  

  return;

}


int avgpool(int8_t* input, float inputScale, int32_t inputOffset, int8_t *output, float outputScale, int32_t outputOffset,
            dim_t inputDim0, dim_t inputDim1, dim_t inputDim2, dim_t inputDim3,
            dim_t outputDim0, dim_t outputDim1, dim_t outputDim2, dim_t outputDim3,
            unsigned_t KH, unsigned_t KW, unsigned_t pad_size, unsigned_t stride_size) {

#if DEBUG_INFO
printf("is:%f, io%d I(%d,%d,%d,%d) os:%f oo:%d I(%d,%d,%d,%d)  K(%d,%d) pad:%d, st:%d\n",
 inputScale, inputOffset, inputDim0, inputDim1,  inputDim2,  inputDim3, 
  outputScale, outputOffset,   
        outputDim0,  outputDim1, outputDim2, outputDim3,  KH,  KW,  pad_size, stride_size);
#endif
    avgpool_nxn_nhwc(input,inputScale, inputOffset, output, outputScale, outputOffset, inputDim0, inputDim1,  inputDim2,  inputDim3,  
        outputDim0,  outputDim1, outputDim2, outputDim3,  KH,  KW,  pad_size, stride_size);
		
/*			
  std::array<unsigned_t,4> padsS = {pad_size, pad_size, pad_size, pad_size};
  llvm::ArrayRef<unsigned_t> padShape(padsS);

  std::array<unsigned_t,2> kernelS = {KH, KW};
  llvm::ArrayRef<unsigned_t> kernelShape(kernelS);

  std::array<unsigned_t,2> strideS = {stride_size, stride_size};
  llvm::ArrayRef<unsigned_t> strideShape(strideS);

  llvm::SmallVector<dim_t, 4> inputDims = {inputDim0, inputDim1, inputDim2, inputDim3};
  Type inputType(ElemKind::Int8QTy, inputDims, inputScale, inputOffset);
  Tensor inputs((void*)input, &inputType);

  llvm::SmallVector<dim_t, 4> outputDims = {outputDim0, outputDim1, outputDim2, outputDim3};
  Type outputType(ElemKind::Int8QTy, outputDims, outputScale, outputOffset);
  Tensor outputs((void*)output, &outputType);

  fwdAvgPoolInstI8Impl(&inputs, &outputs, kernelShape, strideShape, padShape);
  return 0;
*/  
}

int quantize(int8_t *input, int8_t *output, dim_t size, float scale, int32_t offset){
  const float32x4_t       scale_v = vdupq_n_f32( 1.0f / scale);
  assert(size%16==0);
  
  //printf("scale:%f\n",scale);
  float32_t* in = (float32_t*)input;
  for(int i=0;i<size;i+=16)
  {
    //const float32x4x4_t src = vld1q_f32_x4(in + i);
    const float32x4_t s1 = vld1q_f32(in + i);
    const float32x4_t s2 = vld1q_f32(in + i + 4);
    const float32x4_t s3 = vld1q_f32(in + i + 8);
    const float32x4_t s4 = vld1q_f32(in + i + 12);
    //vcvtaq�ݿø��� ���� �ȸ���
    //vcttnq�� �ؾ���. round even to zero
    const int32x4x4_t rf =
    {
        {
            vcvtnq_s32_f32(vmulq_f32(s1, scale_v)),
            vcvtnq_s32_f32(vmulq_f32(s2, scale_v)),
            vcvtnq_s32_f32(vmulq_f32(s3, scale_v)),
            vcvtnq_s32_f32(vmulq_f32(s4, scale_v)),
        }
    };  
    const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
    const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
    vst1q_s8(output + i, vcombine_s8(pa, pb));
  }
  return 0;
}

int dequantize(int8_t *input, int8_t *output, dim_t size, float scale, int32_t offset){
  // output is type float*

  const float32x4_t       scale_v = vdupq_n_f32(scale);
  
  // only for size%8 ==0
  assert(size%8==0);

  float32_t *out = (float32_t*)output;

  for(int i=0;i<size/8;i++)
  {
      const int8x8_t     a = vld1_s8(input + i*8);

      const float32x4x2_t af =
      {
              {
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(a)))), scale_v),
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(a)))),  scale_v),
              }
      };

      //vst1q_f32_x2(out+i*8, af);
      vst1q_f32(out+i*8, af.val[0]);
      vst1q_f32(out+i*8+4, af.val[1]);
  }
  return 0;
}


int transpose(int8_t *input, int8_t *output, dim_t inDim0, dim_t inDim1, dim_t inDim2, dim_t inDim3,
              dim_t outDim0, dim_t outDim1, dim_t outDim2, dim_t outDim3,
              unsigned_t shf0, unsigned_t shf1, unsigned_t shf2, unsigned_t shf3) {

  llvm::SmallVector<dim_t, 4> inputDims = {inDim0, inDim1, inDim2, inDim3};
  Type inputType(ElemKind::Int8QTy, inputDims, 1.0, 0);
  Tensor inputs((void*)input, &inputType);

  llvm::SmallVector<dim_t, 4> outputDims = {outDim0, outDim1, outDim2, outDim3};
  Type outputType(ElemKind::Int8QTy, outputDims, 1.0, 0);
  Tensor outputs((void*)output, &outputType);

  std::array<unsigned_t, 4> T{{shf0, shf1, shf2, shf3}};
  llvm::ArrayRef<unsigned_t> shuffle(T);

  assert(inputs.size() == outputs.size() && "Invalid tensor dimensions");

  inputs.transpose(&outputs, shuffle);

  return 0;
}

int elemadd(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
            int8_t *output, float destScale, int32_t destOffset, unsigned_t size) {
  
  assert(input0Offset==0);
  assert(input1Offset==0);


  //use largeScale to prevent overflow
  const float32x4_t i0scale  = vdupq_n_f32(input0Scale * (1<<15));
  const float32x4_t i1scale  = vdupq_n_f32(input1Scale * (1<<15));
  const float32x4_t invdscale  = vdupq_n_f32((float(1) / ((1<<15)) / destScale));
   
  // clip
  const int32x4_t max = vdupq_n_s32(127);
  const int32x4_t min = vdupq_n_s32(-128);

#if DEBUG_INFO
  printf("lio: %d rio: %d\n", input0Offset, input1Offset);
  printf("largeInv: %f\n",(float(1) / ((1<<15)) / destScale));

#endif

  if(size%16 == 0)
  {
    for(int i=0; i<size; i+=16)
    {
    
      
        const int8x16_t     a = vld1q_s8(input0 + i);
        const int32x4x4_t af =
        {
            {
              // vget_low_s8 = 16���� 8���� ������
              // vmovl_s8 = int8->int16(������ int16, int32�� ����)
              // vget_low_s16 = int16*8 -> int 16*4
              // vmovl_s16 = int16->int32 (float conv�� s32�� ����.)
              // vcctq_f32_s32 = conv s32->f32

                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(a))))), i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(a))))),  i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(a))))),  i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(a))))), i0scale)),
            }
        };
        const int8x16_t     b = vld1q_s8(input1 + i);
        const int32x4x4_t bf =
        {
            {
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(b))))), i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(b))))),  i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(b))))), i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(b))))),  i1scale)),
            }
        };          
        //���� �ڵ尡 ������ roundó�� �� �ڿ� �ٽ� float�� �ٲ� ó���ϱ⶧���� �����ϰ� ó��
        const int32x4x4_t rf =
        {
            {

                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(af.val[0], bf.val[0])), invdscale)), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(af.val[1], bf.val[1])), invdscale)), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(af.val[2], bf.val[2])), invdscale)), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vaddq_s32(af.val[3], bf.val[3])), invdscale)), max),min),
            }
        };
        const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
        const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
        vst1q_s8(output + i, vcombine_s8(pa, pb));

#if DEBUG_INFO

        // load org
        int8_t aa[32],bb[32];
        vst1q_s8(aa,a);
        vst1q_s8(bb,b);     
        for(int k=0;k<16;k++)
        {
          printf("(%d,%d) ", aa[k],bb[k]);
        }
        printf("\n");
        
        int at[4],bt[4];
        int rt[4];

        for(int k=0;k<4;k++)
        {
          vst1q_s32(at, af.val[k]);
          vst1q_s32(bt, bf.val[k]);            
          vst1q_s32(rt, rf.val[k]);  
          for(int j=0;j<4;j++) 
          {
            int idx=i+k*4+j;
            printf("[%d]  l:%d r:%d sum32:%d o:%d \n",idx, at[j], rt[j], rt[j],output[idx]);
          }
        }
#endif          
    }
  }
  return 0;
}

int elemadd_relu(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
            int8_t *output, float destScale, int32_t destOffset, unsigned_t size) {
  for (dim_t i = 0, e = size; i < e; i++) {
    int32_t L = input0[i];
    int32_t R = input1[i];

    // We increase the size of the integer up to 16 bits to prevent overflow.
    const float largeScale = float(1) / (1 << 15);
    // Scale both sides from 8-bit to 16-bits.
    int32_t L32 = std::round(float(L - input0Offset) * (input0Scale / largeScale));
    int32_t R32 = std::round(float(R - input1Offset) * (input1Scale / largeScale));
    int32_t sum32 = L32 + R32;
    sum32 = std::round(float(sum32) * (largeScale / destScale) + destOffset);
    output[i] = quantization::clip<int32_t, int8_t>(sum32);
    if(output[i]<0) output[i]=0;
  }
  return 0;
}


int elemsub(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
            int8_t *output, float destScale, int32_t destOffset, unsigned_t size) {

  assert(input0Offset==0);
  assert(input1Offset==0);
  assert(size%16==0);

  // use largeScale to prevent overflow as elemadd
  const float32x4_t i0scale  = vdupq_n_f32(input0Scale * (1<<15));
  const float32x4_t i1scale  = vdupq_n_f32(input1Scale * (1<<15));
  const float32x4_t invdscale  = vdupq_n_f32((float(1) / ((1<<15)) / destScale));
  
  // clip
  const int32x4_t max = vdupq_n_s32(127);
  const int32x4_t min = vdupq_n_s32(-128);

#if DEBUG_INFO
  printf("lio: %d rio: %d\n", input0Offset, input1Offset);
  printf("largeInv: %f\n",(float(1) / ((1<<15)) / destScale));  
#endif

  if(size%16 == 0)
  {
    for(int i=0; i<size; i+=16)
    {
        const int8x16_t     a = vld1q_s8(input0 + i);
        const int32x4x4_t af =
        {
            {
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(a))))), i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(a))))),  i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(a))))),  i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(a))))), i0scale)),
            }
        };
        const int8x16_t     b = vld1q_s8(input1 + i);
        const int32x4x4_t bf =
        {
            {
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(b))))), i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(b))))),  i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(b))))), i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(b))))),  i1scale)),
            }
        };          
        const int32x4x4_t rf =
        {
            {
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(af.val[0], bf.val[0])), invdscale)), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(af.val[1], bf.val[1])), invdscale)), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(af.val[2], bf.val[2])), invdscale)), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vsubq_s32(af.val[3], bf.val[3])), invdscale)), max),min),
            }
        };
        const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
        const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
        vst1q_s8(output + i, vcombine_s8(pa, pb));
    }
  }
  return 0;
}

int elemdiv(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
            int8_t *output, float destScale, int32_t destOffset, unsigned_t size) {
#if DEBUG_INFO              
  printf("elemdiv: i0s:%f i0o:%d  i1s:%f i1o:%d  ds:%f do:%d \n",input0Scale, input0Offset, input1Scale, input1Offset, destScale, destOffset);
#endif

  assert(input0Offset==0);
  assert(input1Offset==0);
  assert(size%16==0);

  const float32x4_t i0scale  = vdupq_n_f32(input0Scale);
  const float32x4_t i1scale  = vdupq_n_f32(input1Scale * destScale);

  // clip
  const int32x4_t max = vdupq_n_s32(127);
  const int32x4_t min = vdupq_n_s32(-128);

  if(size%16 == 0)
  {
    for(int i=0; i<size; i+=16)
    {
        const int8x16_t     a = vld1q_s8(input0 + i);
        const float32x4x4_t af =
        {
            {
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(a))))), i0scale),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(a))))),  i0scale),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(a))))),  i0scale),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(a))))), i0scale),
            }
        };
        const int8x16_t     b = vld1q_s8(input1 + i);
        const float32x4x4_t bf =
        {
            {
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(b))))), i1scale),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(b))))),  i1scale),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(b))))), i1scale),
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(b))))),  i1scale),
            }
        };          
        //rounding to nearest with ties to away
        const int32x4x4_t rf =
        {
            {
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vdivq_f32(af.val[0], bf.val[0])), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vdivq_f32(af.val[1], bf.val[1])), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vdivq_f32(af.val[2], bf.val[2])), max),min),
                vmaxq_s32(vminq_s32(vcvtaq_s32_f32(vdivq_f32(af.val[3], bf.val[3])), max),min),
            }
        };
        const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
        const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
        vst1q_s8(output + i, vcombine_s8(pa, pb));
    }
  }
  return 0;
}

int splat(int8_t *output, unsigned_t size, int8_t value) {

  //size = n * 16
  assert(size%16==0);

  const int8x16_t v = vdupq_n_s8(value);
  for(int i=0; i<size; i+=16)
  {
    vst1q_s8(output+i, v);
  }
  return 0;
}


int elemmax(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
            int8_t *output, float destScale, int32_t destOffset, unsigned_t size) {

#if DEBUG_INFO
  printf("elemmax: i0s:%f i0o:%d  i1s:%f i1o:%d  ds:%f do:%d \n",input0Scale, input0Offset, input1Scale, input1Offset, destScale, destOffset);    
#endif

  assert(input0Offset==0);
  assert(input1Offset==0);
  assert(size%16==0);

  const float32x4_t i0scale  = vdupq_n_f32(input0Scale / destScale);
  const float32x4_t i1scale  = vdupq_n_f32(input1Scale / destScale);
   
  // for clip s8
  const int32x4_t max = vdupq_n_s32(127);
  const int32x4_t min = vdupq_n_s32(-128);

  //case 1. size = n * 16, offset = 0
  if(size%16 == 0)
  {
    int s = size/16;
    for(int i=0; i<size; i+=16)
    {
        const int8x16_t   a = vld1q_s8(input0 + i);
        const int32x4x4_t af =
        {
            {
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(a))))), i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(a))))),  i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(a))))),  i0scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(a))))), i0scale)),
            }
        };

        const int8x16_t   b = vld1q_s8(input1 + i);
        const int32x4x4_t bf =
        {
            {
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(b))))), i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(b))))),  i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(b))))), i1scale)),
                vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(b))))),  i1scale)),
            }
        };          

        const int32x4x4_t rf =
        {
            {
                vmaxq_s32(vminq_s32(vmaxq_s32(af.val[0], bf.val[0]), max),min),
                vmaxq_s32(vminq_s32(vmaxq_s32(af.val[1], bf.val[1]), max),min),
                vmaxq_s32(vminq_s32(vmaxq_s32(af.val[2], bf.val[2]), max),min),
                vmaxq_s32(vminq_s32(vmaxq_s32(af.val[3], bf.val[3]), max),min),
            }
        };
        const int8x8_t pa = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
        const int8x8_t pb = vqmovn_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
        vst1q_s8(output + i, vcombine_s8(pa, pb));
    }
  }  
  return 0;
}

int softmax(int8_t *input, int8_t *output, dim_t inDim0, dim_t inDim1,
            dim_t outDim0, dim_t outDim1){
  llvm::SmallVector<dim_t, 2> inputDims = {inDim0, inDim1};
  Type inputType(ElemKind::FloatTy, inputDims);
  Tensor inputs((void*)input, &inputType);

  llvm::SmallVector<dim_t, 2> outputDims = {outDim0, outDim1};
  Type outputType(ElemKind::FloatTy, outputDims);
  Tensor outputs((void*)output, &outputType);

  auto inW = inputs.getHandle<float>();
  auto outW = outputs.getHandle<float>();
  auto idim = inW.dims();

  for (dim_t n = 0; n < idim[0]; n++) {
    // Find Max.
    float max = float(inW.at({n, 0}));
    for (dim_t i = 1; i < idim[1]; i++) {
      max = std::max(max, float(inW.at({n, i})));
    }

    // Compute exp.
    float sum = 0;
    for (dim_t i = 0; i < idim[1]; i++) {
      float e = std::exp(float(inW.at({n, i})) - max);
      sum += e;
      outW.at({n, i}) = float(e);
    }
    // Normalize the output.
    for (dim_t i = 0; i < idim[1]; i++) {
      outW.at({n, i}) = float(outW.at({n, i})) / sum;
    }
  } // N

  return 0;
}


void debugprint(int8_t *input, unsigned_t size, std::string filename, bool isInt8 ) {
  std::ofstream fos(filename.c_str(), std::ios::binary);

  if(isInt8) {
    int16_t data16 = 0;
    for (size_t i = 0, e = size; i < e; i++) {
      auto data = input[i];
      if (i % 2 == 0) {
        data16 = 0xff & data;
      } else {
        data16 = data16 | data << 8;
        fos.write((const char *) &data16, 2);
      }
    }
  }
  else{
    int32_t* input_32 = (int32_t*) input;
    for (size_t i = 0, e = size; i < e; i++) {
      auto data = input_32[i];
      fos.write((const char *)&data, 4);
    }
  }
  fos.close();
}


void transpose_nhwc2vtaio(int8_t* input, int8_t* output, int N, int H, int W, int C){
  int8_t *IN_NHWC = input;

  int8_t *IN_NCHW = (int8_t *)malloc(N*C*H*W);
  /*
  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          *(IN_NCHW + n * C * H * W + c * H * W + h * W + w) =
              *(IN_NHWC + n * H * W * C + h * W * C + w * C + c);
        }
      }
    }
  }
  */
 nhwc_to_nchw_s8(input, IN_NCHW, N,H,W,C);


  // reshape input NCHW -> N{C//16}16HW
  int8_t *in_NC_16HW = IN_NCHW;


  for (int n = 0; n < N; n++) {
    for (int c_ = 0; c_ < C / 16; c_++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          for (int c = 0; c < 16; c++) {
            *(output + n*(C/16)*H*W*16 + c_*H*W*16 + h*W*16 + w*16 + c) =
                *(in_NC_16HW + n*(C/16)*16*H*W + c_*16*H*W + c*H*W + h*W + w);
          }
        }
      }
    }
  }

  free(IN_NCHW);
}

void transpose_nhwc2vtak(int8_t* input, int8_t* output, int KN, int KH, int KW, int C){
  // filter NHWC
  int8_t *filter_NHWC = input;

  int8_t* filter_N_16HWC_16 = filter_NHWC;
  // transpose filter {N//16}16HW{C//16}16 =>  {N//16}{C//16}HW1616
  for (int n_ = 0; n_ < KN / 16; n_++) {
    for (int c_ = 0; c_ < C / 16; c_++) {
      for (int h = 0; h < KH; h++) {
        for (int w = 0; w < KW; w++) {
          for (int n = 0; n < 16; n++) {
            for (int c = 0; c < 16; c++) {
              *(output + n_*(C/16)*KH*KW*16*16 + c_ *KH*KW*16*16 + h*KW*16*16 + w*16*16 + n*16 + c) =
                  *(filter_N_16HWC_16 + n_*16*KH*KW*C + n*KH*KW*C + h*KW*C + w*C + c_*16 + c);
            }
          }
        }
      }
    }
  }
}

void transpose_vtak2nhwc(int8_t* input, int8_t* output, int KN, int KH, int KW, int C){
  for (int n_ = 0; n_ < KN / 16; n_++) {
    for (int c_ = 0; c_ < C / 16; c_++) {
      for (int h = 0; h < KH; h++) {
        for (int w = 0; w < KW; w++) {
          for (int n = 0; n < 16; n++) {
            for (int c = 0; c < 16; c++) {
              *(output + n_*16*KH*KW*C + n*KH*KW*C + h*KW*C + w*C + c_*16 + c)
                  = *(input + n_*(C/16)*KH*KW*16*16 + c_ *KH*KW*16*16 + h*KW*16*16 + w*16*16 + n*16 + c);
            }
          }
        }
      }
    }
  }
}


void transpose_vtaio2nhwc(int8_t* input, int8_t* output, int N, int H, int W, int C){
  for (int b = 0; b < N; b++) {
    for (int f = 0; f < C; f++) {
      for (int i = 0; i < H; i++) {    //h
        for (int j = 0; j < W; j++) {    //w

          int out_ch = f / 16;
          int oo = f % 16;
          auto value =*(input + (out_ch * H * W * 16 + i * W * 16 + j * 16 + oo));
          *(output + b * H * W * C + i * W * C + j * C + f) = value;
        }
      }
    }
  }
}