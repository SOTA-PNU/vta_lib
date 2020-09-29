#ifndef VTALIB_EXAMPLE_VTABUNDLE_H
#define VTALIB_EXAMPLE_VTABUNDLE_H

#define dim_t uint64_t
#define unsigned_t uint32_t

#include <string>

void xlnk_reset();

int convolution(int8_t* input, int8_t *kernel, int8_t *output, int N, int H, int W, int C, int KN, int KH, int KW, int pad_size,
                int stride_size, bool doRelu, bool doBias, uint8_t shift, int out_h, int out_w, int tile_h=14, int tile_w=14);

int convolution_wo_tr(int8_t* input, int8_t *kernel, int32_t *bias, int8_t *output, int N, int H, int W, int C, int KN, int KH, int KW, int pad_size,
                       int stride_size, bool doRelu, bool doBias, uint8_t shift, int out_h, int out_w, void* vtaCmdH, int nVirtualThread = 1, int tileHSize = 14, int tileWSize = 14);

int convolutionFloat(int8_t* input, int8_t *kernel, int8_t *bias, int8_t *output, dim_t N, dim_t H, dim_t W, dim_t C,
        dim_t KN, dim_t KH, dim_t KW, int pad_size, int stride_size, size_t group, size_t dilation,
        bool doRelu, bool doBias, int out_h, int out_w);

int nonvtaconvolution(int8_t* input, float inputScale, int32_t inputOffset, int8_t *kernel, float kernelScale, int32_t kernelOffset,
                      int8_t *bias, float biasScale, int32_t biasOffset, int8_t *output, float outputScale, int32_t outputOffset, dim_t N, dim_t H, dim_t W, dim_t C, dim_t KN, dim_t KH, dim_t KW, int pad_size,
                      int stride_size, size_t group, size_t dilation, bool doRelu, bool doBias, int out_h, int out_w);

int fullyconnected(int8_t* input, float inputScale, int32_t inputOffset, int8_t *kernel, float kernelScale, int32_t kernelOffset,
                   int8_t *bias, float biasScale, int32_t biasOffset, int8_t *output, float outputScale, int32_t outputOffset,
                   dim_t inputDim0, dim_t inputDim1, dim_t kernelDim0, dim_t kernelDim1, dim_t outputDim0, dim_t outputDim1, bool doBias);

int maxpool(int8_t* input, int8_t *output, uint64_t N, uint64_t H, uint64_t W, uint64_t C, uint32_t KH, uint32_t KW, uint32_t pad_size,
            uint32_t stride_size);

int avgpool(int8_t* input, float inputScale, int32_t inputOffset, int8_t *output, float outputScale, int32_t outputOffset,
            dim_t inputDim0, dim_t inputDim1, dim_t inputDim2, dim_t inputDim3,
            dim_t outputDim0, dim_t outputDim1, dim_t outputDim2, dim_t outputDim3,
            unsigned_t KH, unsigned_t KW, unsigned_t pad_size, unsigned_t stride_size);

int relu(int8_t* input, int8_t *output, unsigned_t size);

int quantize(int8_t *input, int8_t *output, uint64_t size, float scale, int32_t offset);

int dequantize(int8_t *input, int8_t *output, uint64_t size, float scale, int32_t offset);

int transpose(int8_t *input, int8_t *output, dim_t inDim0, dim_t inDim1, dim_t inDim2, dim_t inDim3,
              dim_t outDim0, dim_t outDim1, dim_t outDim2, dim_t outDim3,
              unsigned_t shf0, unsigned_t shf1, unsigned_t shf2, unsigned_t shf3);

int elemadd(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
            int8_t *output, float destScale, int32_t destOffset, unsigned_t size);

int elemsub(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
            int8_t *output, float destScale, int32_t destOffset, unsigned_t size);

int elemdiv(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
            int8_t *output, float destScale, int32_t destOffset, unsigned_t size);

int splat(int8_t *input, unsigned_t size, int8_t value);

int elemmax(int8_t *input0, float input0Scale, int32_t input0Offset, int8_t *input1, float input1Scale, int32_t input1Offset,
             int8_t *output, float destScale, int32_t destOffset, unsigned_t size);

int softmax(int8_t *input, int8_t *output, dim_t inDim0, dim_t inDim1,
            dim_t outDim0, dim_t outDim1);

void debugprint(int8_t *input, unsigned_t size, std::string filename, bool isInt8 );

void transpose_nhwc2vtaio(int8_t* input, int8_t* output, int N, int H, int W, int C);

void transpose_nhwc2vtak(int8_t* input, int8_t* output, int KN, int KH, int KW, int C);

void transpose_vtaio2nhwc(int8_t* input, int8_t* output, int N, int H, int W, int C);
#endif //VTALIB_EXAMPLE_VTABUNDLE_H
