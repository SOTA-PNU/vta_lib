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

#if(GLOW_WITH_VTASIM)
#include "vta/vtalib/include/simulator/runtime.h"
#else
#include "vta/vtalib/include/zcu102/vta/runtime.h"
#endif
#include "vta/vtalib/include/Bundle/VTABundle.h"

#define RESET_IOCTL _IOWR('X', 101, unsigned long)

using namespace std;
using namespace glow;

#define MIN_W_SIZE 7
#define MIN_H_SIZE 7
#define OUT_TILE_SIZE_W 14
#define OUT_TILE_SIZE_H 14


void xlnk_reset() {
#if(!(GLOW_WITH_VTASIM==1))
  int xlnkfd = open("/dev/xlnk", O_RDWR | O_CLOEXEC);
  if (xlnkfd < 0) {
    printf("Reset failed - could not open device: %d\n", xlnkfd);
    return;
  }
  if (ioctl(xlnkfd, RESET_IOCTL, 0) < 0) {
    printf("Reset failed - IOCTL failed: %d\n", errno);
  }
  close(xlnkfd);
#endif
}

struct ResetAcc {
  uint32_t base_;
  uint32_t size_;
};

int reset_acc(void *param) {
  ResetAcc *p = (ResetAcc *) (param);
  VTAUopLoopBegin(p->size_, 1, 0, 0);
  VTAUopPush(0, 1, p->base_, 0, 0, 0, 0, 0);
  VTAUopLoopEnd();
  return 0;
}

struct gemmop_mxn {
  uint32_t tile_size_h_;
  uint32_t tile_size_w_;
  uint32_t in_tile_size_h_;
  uint32_t in_tile_size_w_;
  uint32_t src_factor_;
  uint32_t kernel_h_;
  uint32_t kernel_w_;
  uint32_t stride_h_;
  uint32_t stride_w_;
  uint32_t n_kernel_;
};
int gemmop_mxn(void *param) {

  struct gemmop_mxn *ctx = (struct gemmop_mxn *) (param);
  VTAUopLoopBegin(ctx->tile_size_h_, ctx->tile_size_w_, ctx->src_factor_ * ctx->stride_h_, 0);
  VTAUopLoopBegin(ctx->kernel_h_, 0, ctx->src_factor_, ctx->kernel_w_);

  for (int dx = 0; dx < ctx->kernel_w_; dx++) {
    for (int j = 0; j < ctx->tile_size_w_; j++) {
      VTAUopPush(0, 0, j, j*ctx->stride_w_ + dx, dx, 0, 0, 0);
    }
  }
  VTAUopLoopEnd();
  VTAUopLoopEnd();

  return 0;
}

int gemmop_mxn2(void *param) {

  struct gemmop_mxn *ctx = (struct gemmop_mxn *) (param);
  VTAUopLoopBegin(ctx->tile_size_w_, 1, ctx->stride_w_, 0);
  VTAUopLoopBegin(ctx->kernel_w_, 0, 1, 1);

  for (int dx = 0; dx < ctx->kernel_h_; dx++) {
    for (int kernelIdx = 0; kernelIdx < ctx->n_kernel_; kernelIdx++) {
      for (int inTileHIdx = 0; inTileHIdx < ctx->tile_size_h_; inTileHIdx++) {
        auto outTileHIdx = inTileHIdx + (kernelIdx * ctx->tile_size_h_);
        VTAUopPush(0, 0, outTileHIdx * ctx->tile_size_w_,
                   (inTileHIdx * ctx->stride_h_ + dx) * (ctx->in_tile_size_w_), dx * ctx->kernel_w_ + kernelIdx * (ctx->kernel_w_ * ctx->kernel_h_), 0, 0,
                   0);
      }
    }
  }
  VTAUopLoopEnd();
  VTAUopLoopEnd();

  return 0;
}

// alu func
struct ShiftContext {
  uint32_t base_;
  uint32_t size_;
  uint8_t shift_;
};
int alu_shr(void *param) {
  ShiftContext *p = (ShiftContext *) (param);
  VTAUopLoopBegin(p->size_, 1, 1, 0);
  VTAUopPush(1, 0, p->base_, 0, 0, 3, 1, p->shift_);
  VTAUopLoopEnd();
  return 0;
}


struct AddContext {
  uint32_t dst_;
  uint32_t src_;
  uint32_t size_;
  uint32_t iter_;
};

int alu_add(void *param) {
  AddContext *p = (AddContext *) (param);
  VTAUopLoopBegin(p->iter_, p->size_, 1, 0);
  VTAUopLoopBegin(p->size_, 1, 0, 0);

  VTAUopPush(1, 0, p->dst_, p->src_, 0, 2, 0, 0);

  VTAUopLoopEnd();
  VTAUopLoopEnd();
  return 0;
}
int alu_max(void *param) {
  ResetAcc *p = (ResetAcc *) (param);
  VTAUopLoopBegin(p->size_, 1, 1, 0);
  VTAUopPush(1, 0, p->base_, 0, 0, 1, 1, -128);
  VTAUopLoopEnd();
  return 0;
}

int alu_relu(void *param) {
  ResetAcc *p = (ResetAcc *) (param);
  VTAUopLoopBegin(p->size_, 1, 1, 0);
  VTAUopPush(1, 0, p->base_, 0, 0, 1, 1, 0);
  VTAUopLoopEnd();
  return 0;
}

int alu_min(void *param) {
  ResetAcc *p = (ResetAcc *) (param);
  VTAUopLoopBegin(p->size_, 1, 1, 0);
  VTAUopPush(1, 0, p->base_, 0, 0, 0, 1, 127);
  VTAUopLoopEnd();
  return 0;
}

void conv_kernel(uint32_t sizeOutTileH,
                 uint32_t sizeOutTileW,
                 uint32_t pad_size,
                 uint32_t stride_size,
                 uint32_t x_pad_before,
                 uint32_t x_pad_after,
                 uint32_t y_pad_before,
                 uint32_t y_pad_after,
                 uint32_t outputWOffset,
                 uint32_t outputHOffset,
                 uint32_t sizeInTileH,
                 uint32_t sizeInTileW,
                 uint32_t numInputChannel,
                 uint32_t inputHStride,
                 uint32_t inputWStride,
                 void *vta_input_buf,
                 uint32_t KN,
                 uint32_t KH,
                 uint32_t KW,
                 void *vta_filter_buf,
                 uint32_t out_h,
                 uint32_t out_w,
                 void *vta_output_buf,
                 bool relu,
                 uint8_t shift,
                 bool bias,
                 void *vta_bias_buf,
                 VTACommandHandle& vtaCmdH

) {
  void *uopHandle_gemm[2];
  uopHandle_gemm[0] = nullptr;
  void *uopHandle_add[2];
  uopHandle_add[0] = nullptr;
  void *uopHandle_shr[2];
  uopHandle_shr[0] = nullptr;
  void *uopHandle_max[2];
  uopHandle_max[0] = nullptr;
  void *uopHandle_min[2];
  uopHandle_min[0] = nullptr;
  uint32_t resizedOutTileH = (sizeInTileH-KH) / stride_size + 1;
  uint32_t resizedOutTileW = (sizeInTileW-KW) / stride_size + 1;
  uint32_t resizedInTileH =  (resizedOutTileH - 1) * stride_size + KH;
  uint32_t resizedInTileW =  (resizedOutTileW - 1) * stride_size + KW;

  uint32_t nFilterinLoop = KN / 16;
  nFilterinLoop = min(nFilterinLoop, 2048 / resizedOutTileH / resizedOutTileW);
  for(uint32_t i = 31; i>=0; i--){
    if(nFilterinLoop & 0x1 <<i) {
      nFilterinLoop = 0x1 <<i;
      break;
    }
  }
  for (uint32_t filterIdx = 0; filterIdx < KN / nFilterinLoop /16; filterIdx++) {

    uint32_t outDummyH = 0;
    uint32_t inDummyH = 0;
    uint32_t outDummyW = 0;
    uint32_t inDummyW = 0;



    if(resizedOutTileH < MIN_H_SIZE)
    {
      outDummyH = MIN_H_SIZE - resizedOutTileH;
      resizedOutTileH = MIN_H_SIZE;
      resizedInTileH = (resizedOutTileH - 1) * stride_size + KH;
      inDummyH = resizedInTileH - sizeInTileH;
    }

    if(resizedOutTileW < MIN_W_SIZE)
    {
      outDummyW = MIN_W_SIZE - resizedOutTileW;
      resizedOutTileW = MIN_W_SIZE;
      resizedInTileW = (resizedOutTileW - 1) * stride_size + KW;
      inDummyW = resizedInTileW - sizeInTileW;

    }

// acc buffer reset
    void *uopHandle_reset[2];
    uopHandle_reset[0] = nullptr;

    ResetAcc reset;
    reset.base_ = 0;
    reset.size_ = resizedOutTileH * resizedOutTileW * nFilterinLoop;
    VTAPushGEMMOp(uopHandle_reset, &reset_acc, &reset, 0);

    uint32_t inputWOffset = (int) (outputWOffset * stride_size - pad_size ) < 0 ? 0 : outputWOffset * stride_size - pad_size;
    uint32_t inputHOffset = (int) (outputHOffset * stride_size - pad_size ) < 0 ? 0 : outputHOffset * stride_size - pad_size;

    for (uint32_t channelIdx = 0; channelIdx < numInputChannel / 16; channelIdx++) {
      uint32_t sram_offset = channelIdx * inputHStride * inputWStride + inputHOffset * inputWStride + inputWOffset;
      uint32_t filter_offset = filterIdx * nFilterinLoop * (numInputChannel/16) * KH * KW + channelIdx * KH * KW;
      uint32_t y_size = (inputHStride - inputHOffset + y_pad_before) < resizedInTileH ?
                        inputHStride - inputHOffset : resizedInTileH - y_pad_before;
      uint32_t x_size = (inputWStride - inputWOffset + x_pad_before) < resizedInTileW ?
                        inputWStride - inputWOffset : resizedInTileW - x_pad_before;

      VTADepPush(vtaCmdH, 2, 1);
      VTADepPop(vtaCmdH, 2, 1);
      VTALoadBuffer2D(vtaCmdH, vta_input_buf,
                      sram_offset,
                      x_size,
                      y_size,
                      inputWStride,
                      x_pad_before,
                      y_pad_before,
                      resizedInTileW - x_pad_before - x_size,
                      resizedInTileH - y_pad_before - y_size,
                      0,        //index
                      2);    //type

      VTALoadBuffer2D(vtaCmdH, vta_filter_buf,
                      filter_offset,
                      KH * KW,
                      nFilterinLoop,
                      KH * KW * numInputChannel / 16,
                      0, 0, 0, 0, 0, 1);

      VTADepPush(vtaCmdH, 1, 2);
      VTADepPop(vtaCmdH, 1, 2);



      struct gemmop_mxn gemm_ctx;
      gemm_ctx.tile_size_h_ = resizedOutTileH;
      gemm_ctx.tile_size_w_ = resizedOutTileW;
      gemm_ctx.src_factor_ = resizedInTileW;
      gemm_ctx.in_tile_size_h_ = resizedInTileH;
      gemm_ctx.in_tile_size_w_ = resizedInTileW;
      gemm_ctx.kernel_h_ = KH;
      gemm_ctx.kernel_w_ = KW;
      gemm_ctx.stride_h_ = stride_size;
      gemm_ctx.stride_w_ = stride_size;
      gemm_ctx.n_kernel_ = nFilterinLoop;

      VTAPushGEMMOp(uopHandle_gemm, &gemmop_mxn2, &gemm_ctx, 0);



    }

    if(bias){

      uint32_t sram_offset = resizedOutTileH * resizedOutTileW * nFilterinLoop;
      VTALoadBuffer2D(vtaCmdH, (int32_t*)vta_bias_buf,
                      filterIdx * nFilterinLoop,
                      16* nFilterinLoop,
                      1,
                      16* nFilterinLoop,
                      0,
                      0,
                      0,
                      0,
                      sram_offset,        //index
                      3);    //type



      AddContext add_ctx;
      add_ctx.dst_ = 0;
      add_ctx.size_ = resizedOutTileH * resizedOutTileW;
      add_ctx.src_ = resizedOutTileH * resizedOutTileW * nFilterinLoop;
      add_ctx.iter_ = nFilterinLoop;
      VTAPushALUOp(uopHandle_add, &alu_add, &add_ctx, 0);


    }



    ShiftContext shift_ctx;
    shift_ctx.base_ = 0;
    shift_ctx.size_ = resizedOutTileH * resizedOutTileW * nFilterinLoop;
    shift_ctx.shift_ = shift;


    VTAPushALUOp(uopHandle_shr, &alu_shr, &shift_ctx, 0);



    if(!relu) {
      VTAPushALUOp(uopHandle_max, &alu_max, &reset, 0);
    }
    else{
      VTAPushALUOp(uopHandle_max, &alu_relu, &reset, 0);
    }

    VTAPushALUOp(uopHandle_min, &alu_min, &reset, 0);


    VTADepPush(vtaCmdH, 2, 3);
    VTADepPop(vtaCmdH, 2, 3);

    if (outDummyW == 0) {
      for(uint32_t fIdx = 0; fIdx < nFilterinLoop; fIdx++) {
        VTAStoreBuffer2D(vtaCmdH, fIdx * resizedOutTileH * resizedOutTileW, 4, vta_output_buf,
                         filterIdx * nFilterinLoop * out_w * out_h +
                         outputHOffset * out_w + outputWOffset + fIdx * out_w * out_h,
                         sizeOutTileW, sizeOutTileH, out_w);
      }
    } else {
      for (int hIdx = 0; hIdx < sizeOutTileH * nFilterinLoop; hIdx++) {
        VTAStoreBuffer2D(vtaCmdH,
                         0 + hIdx * resizedOutTileW,
                         4,
                         vta_output_buf,
                         filterIdx * nFilterinLoop * out_w * out_h+(outputHOffset + hIdx) * out_w + outputWOffset,
                         sizeOutTileW,
                         1,
                         out_w);
      }
    }
    VTADepPush(vtaCmdH, 3, 2);
    VTADepPop(vtaCmdH, 3, 2);
  }
}



int convolution(int8_t* input, int8_t *kernel, int8_t *output, int N, int H, int W, int C, int KN, int KH, int KW, int pad_size,
                int stride_size, bool doRelu, bool doBias, uint8_t shift, int out_h, int out_w, int tile_h, int tile_w) {
  int8_t *IN_NHWC = input;

  //transpose input NHWC -> NCHW
  int8_t *IN_NCHW = (int8_t *)malloc(N*C*H*W);
  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          *(IN_NCHW + n*C*H*W + c*H*W + h*W + w)=*(IN_NHWC + n*H*W*C + h*W*C + w*C + c);
        }
      }
    }
  }

  // reshape input NCHW -> N{C//16}16HW
  int8_t *in_NC_16HW = IN_NCHW;

  //transpose input N{C//16}16HW -> N{C//16}HW16
  int8_t *in_NC_HW16 = (int8_t *)malloc(N * (C / 16) * H * W * 16);
  for (int n = 0; n < N; n++) {
    for (int c_ = 0; c_ < C / 16; c_++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          for (int c = 0; c < 16; c++) {
            *(in_NC_HW16 + n*(C/16)*H*W*16 + c_*H*W*16 + h*W*16 + w*16 + c) =
                    *(in_NC_16HW + n*(C/16)*16*H*W + c_*16*H*W + c*H*W + h*W + w);
          }
        }
      }
    }
  }

  // filter NHWC
  int8_t *filter_NHWC = kernel;

  // reshape filter NHWC -> {N//16}16HW{C//16}16
  assert(KN % 16 == 0);

  int8_t* filter_N_16HWC_16 = filter_NHWC;

  // transpose filter {N//16}16HW{C//16}16 =>  {N//16}{C//16}HW1616
  int8_t* filter_N_C_HW1616 = (int8_t*) malloc ((KN / 16) *(C / 16)*KH*KW*16*16);
  for (int n_ = 0; n_ < KN / 16; n_++) {
    for (int c_ = 0; c_ < C / 16; c_++) {
      for (int h = 0; h < KH; h++) {
        for (int w = 0; w < KW; w++) {
          for (int n = 0; n < 16; n++) {
            for (int c = 0; c < 16; c++) {
              *(filter_N_C_HW1616 + n_*(C/16)*KH*KW*16*16 + c_ *KH*KW*16*16 + h*KW*16*16 + w*16*16 + n*16 + c) =
                      *(filter_N_16HWC_16 + n_*16*KH*KW*C + n*KH*KW*C + h*KW*C + w*C + c_*16 + c);
            }
          }
        }
      }
    }
  }
  int32_t *bias;
  if(doBias) {
    auto locBias = KN * KW * KH * C;
    assert(locBias % 4 == 0);
    locBias /= 4;

    bias = (int32_t *) kernel + locBias;
  }
  int8_t *out_VTA = (int8_t *) malloc(out_h * out_w * KN);



  // VTA DRAM alloc
  // vta's input buffer, as the same size of in_NC_HW16[N][C/16][H][W][16];
  void *vta_input_buf = VTABufferAlloc(N * C * H * W);
  // vta's filter buffer, as the same size of filter_N_C_HW1616[KN/16][C/16][KH][KW][16][16];
  void *vta_filter_buf = VTABufferAlloc(KN * KH * KW * C);
  void *vta_output_buf = VTABufferAlloc(out_h * out_w * KN);
  void *vta_bias_buf;
  if(doBias) {
    vta_bias_buf = VTABufferAlloc(KN * 4);
  }
  // VTA Buffer copy
  VTABufferCopy(in_NC_HW16, 0, vta_input_buf, 0, N * C * H * W, 1);
  VTABufferCopy(filter_N_C_HW1616, 0, vta_filter_buf, 0, KN * KH * KW * C, 1);

  if(doBias) {
    VTABufferCopy(bias, 0, vta_bias_buf, 0, KN * 4, 1);
  }

  uint32_t nTileW, nTileH;
  uint32_t sizeOutTileW = tile_w;
  uint32_t sizeOutTileH = tile_h;

  uint32_t sizeInTileW = (sizeOutTileW - 1) * stride_size + KW;
  uint32_t sizeInTileH = (sizeOutTileH - 1) * stride_size + KH;

  int isRemainTileW = (out_w % sizeOutTileW) ? 1 : 0;
  int isRemainTileH = (out_h % sizeOutTileH) ? 1 : 0;
  nTileW = out_w / sizeOutTileW + isRemainTileW;
  nTileH = out_h / sizeOutTileH + isRemainTileH;


  VTACommandHandle vtaCmdH{nullptr};
  vtaCmdH = VTATLSCommandHandle();
  //VTASetDebugMode(vtaCmdH, VTA_DEBUG_DUMP_INSN | VTA_DEBUG_DUMP_UOP);
  VTASetDebugMode(vtaCmdH, 0);

  for (int tileH = 0; tileH < nTileH; tileH++) {
    for (int tileW = 0; tileW < nTileW; tileW++) {
      uint32_t x_pad_before = tileW == 0 ? pad_size : 0;
      uint32_t x_pad_after = tileW == (nTileW - 1) ? pad_size : 0;
      uint32_t y_pad_before = tileH == 0 ? pad_size : 0;
      uint32_t y_pad_after = tileH == (nTileH - 1) ? pad_size : 0;
      uint32_t outputWOffset = tileW * sizeOutTileW;
      uint32_t outputHOffset = tileH * sizeOutTileH;

      uint32_t resizedOutTileH = sizeOutTileH;
      uint32_t resizedInTileH = sizeInTileH;
      if (tileH == nTileH-1) {
        resizedOutTileH = out_h - tileH * sizeOutTileH;
        resizedInTileH = (resizedOutTileH -1) * stride_size + KH;
      }

      if ((tileW != nTileW - 1)) {
        conv_kernel(resizedOutTileH,
                    sizeOutTileW,
                    pad_size,
                    stride_size,
                    x_pad_before,
                    x_pad_after,
                    y_pad_before,
                    y_pad_after,
                    outputWOffset,
                    outputHOffset,
                    resizedInTileH,
                    sizeInTileW,
                    C,
                    H,
                    W,
                    vta_input_buf,
                    KN,
                    KH,
                    KW,
                    vta_filter_buf,
                    out_h,
                    out_w,
                    vta_output_buf,
                    doRelu,
                    shift,
                    doBias,
                    vta_bias_buf,
                    vtaCmdH
        );
      } else {
        uint32_t sizeOutTileBoundaryW = out_w - tileW * sizeOutTileW;
        uint32_t sizeInTileBoundaryW = (sizeOutTileBoundaryW - 1) * stride_size + KW ;

        conv_kernel(resizedOutTileH,
                    sizeOutTileBoundaryW,
                    pad_size,
                    stride_size,
                    x_pad_before,
                    x_pad_after,
                    y_pad_before,
                    y_pad_after,
                    outputWOffset,
                    outputHOffset,
                    resizedInTileH,
                    sizeInTileBoundaryW,
                    C,
                    H,
                    W,
                    vta_input_buf,
                    KN,
                    KH,
                    KW,
                    vta_filter_buf,
                    out_h,
                    out_w,
                    vta_output_buf,
                    doRelu,
                    shift,
                    doBias,
                    vta_bias_buf,
                    vtaCmdH
        );

      }
    }
  }
  VTASynchronize(vtaCmdH, 1 << 31);
  VTARuntimeShutdown();

  VTABufferCopy(vta_output_buf, 0, out_VTA, 0, out_h * out_w * KN, 2);

  // end case
  VTABufferFree(vta_input_buf);
  VTABufferFree(vta_output_buf);
  VTABufferFree(vta_filter_buf);
  if(doBias) {
    VTABufferFree(vta_bias_buf);
  }









  for (int b = 0; b < N; b++) {
    for (int f = 0; f < KN; f++) {
      for (int i = 0; i < out_h; i++) {    //h
        for (int j = 0; j < out_w; j++) {    //w

          int out_ch = f / 16;
          int oo = f % 16;
          auto value =*(out_VTA + (out_ch * out_h * out_w * 16 + i * out_w * 16 + j * 16 + oo));
          *(output + b * out_h * out_w * KN + i * out_w * KN + j * KN + f) = value;


        }
      }
    }
  }

  free(out_VTA);

  free(IN_NCHW);
  free(in_NC_HW16);
  free(filter_N_C_HW1616);

  return 0;
}


int convolution_wo_tr(int8_t* input, int8_t *kernel, int32_t *bias, int8_t *output, int N, int H, int W, int C, int KN, int KH, int KW, int pad_size,
                int stride_size, bool doRelu, bool doBias, uint8_t shift, int out_h, int out_w) {
  int8_t *in_NC_HW16 = input;
  int8_t* filter_N_C_HW1616 = kernel;

  // VTA DRAM alloc
  // vta's input buffer, as the same size of in_NC_HW16[N][C/16][H][W][16];
  void *vta_input_buf = VTABufferAlloc(N * C * H * W);
  // vta's filter buffer, as the same size of filter_N_C_HW1616[KN/16][C/16][KH][KW][16][16];
  void *vta_filter_buf = VTABufferAlloc(KN * KH * KW * C);
  void *vta_output_buf = VTABufferAlloc(out_h * out_w * KN);
  void *vta_bias_buf;
  if(doBias) {
    vta_bias_buf = VTABufferAlloc(KN * 4);
  }
  // VTA Buffer copy
  VTABufferCopy(in_NC_HW16, 0, vta_input_buf, 0, N * C * H * W, 1);
  VTABufferCopy(filter_N_C_HW1616, 0, vta_filter_buf, 0, KN * KH * KW * C, 1);

  if(doBias) {
    VTABufferCopy(bias, 0, vta_bias_buf, 0, KN * 4, 1);
  }

  uint32_t nTileW, nTileH;
  uint32_t sizeOutTileW = OUT_TILE_SIZE_W;
  uint32_t sizeOutTileH = OUT_TILE_SIZE_H;

  uint32_t sizeInTileW = (sizeOutTileW - 1) * stride_size + KW;
  uint32_t sizeInTileH = (sizeOutTileH - 1) * stride_size + KH;

  int isRemainTileW = (out_w % sizeOutTileW) ? 1 : 0;
  int isRemainTileH = (out_h % sizeOutTileH) ? 1 : 0;
  nTileW = out_w / sizeOutTileW + isRemainTileW;
  nTileH = out_h / sizeOutTileH + isRemainTileH;


  VTACommandHandle vtaCmdH{nullptr};
  vtaCmdH = VTATLSCommandHandle();
  //VTASetDebugMode(vtaCmdH, VTA_DEBUG_DUMP_INSN | VTA_DEBUG_DUMP_UOP);
  VTASetDebugMode(vtaCmdH, 0);

  for (int tileH = 0; tileH < nTileH; tileH++) {
    for (int tileW = 0; tileW < nTileW; tileW++) {
      uint32_t x_pad_before = tileW == 0 ? pad_size : 0;
      uint32_t x_pad_after = tileW == (nTileW - 1) ? pad_size : 0;
      uint32_t y_pad_before = tileH == 0 ? pad_size : 0;
      uint32_t y_pad_after = tileH == (nTileH - 1) ? pad_size : 0;
      uint32_t outputWOffset = tileW * sizeOutTileW;
      uint32_t outputHOffset = tileH * sizeOutTileH;

      uint32_t resizedOutTileH = sizeOutTileH;
      uint32_t resizedInTileH = sizeInTileH;
      if (tileH == nTileH-1) {
        resizedOutTileH = out_h - tileH * sizeOutTileH;
        resizedInTileH = (resizedOutTileH -1) * stride_size + KH;
      }

      if ((tileW != nTileW - 1)) {
        conv_kernel(resizedOutTileH,
                    sizeOutTileW,
                    pad_size,
                    stride_size,
                    x_pad_before,
                    x_pad_after,
                    y_pad_before,
                    y_pad_after,
                    outputWOffset,
                    outputHOffset,
                    resizedInTileH,
                    sizeInTileW,
                    C,
                    H,
                    W,
                    vta_input_buf,
                    KN,
                    KH,
                    KW,
                    vta_filter_buf,
                    out_h,
                    out_w,
                    vta_output_buf,
                    doRelu,
                    shift,
                    doBias,
                    vta_bias_buf,
                    vtaCmdH
        );
      } else {
        uint32_t sizeOutTileBoundaryW = out_w - tileW * sizeOutTileW;
        uint32_t sizeInTileBoundaryW = (sizeOutTileBoundaryW - 1) * stride_size + KW ;

        conv_kernel(resizedOutTileH,
                    sizeOutTileBoundaryW,
                    pad_size,
                    stride_size,
                    x_pad_before,
                    x_pad_after,
                    y_pad_before,
                    y_pad_after,
                    outputWOffset,
                    outputHOffset,
                    resizedInTileH,
                    sizeInTileBoundaryW,
                    C,
                    H,
                    W,
                    vta_input_buf,
                    KN,
                    KH,
                    KW,
                    vta_filter_buf,
                    out_h,
                    out_w,
                    vta_output_buf,
                    doRelu,
                    shift,
                    doBias,
                    vta_bias_buf,
                    vtaCmdH
        );
      }
    }
  }
  VTASynchronize(vtaCmdH, 1 << 31);
  VTARuntimeShutdown();

  VTABufferCopy(vta_output_buf, 0, output, 0, out_h * out_w * KN, 2);


  // end case
  VTABufferFree(vta_output_buf);

  VTABufferFree(vta_input_buf);
  VTABufferFree(vta_filter_buf);
  if(doBias) {
    VTABufferFree(vta_bias_buf);
  }
  return 0;
}
