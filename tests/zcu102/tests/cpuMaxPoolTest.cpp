
#include <iostream>
#include <ctime>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>



#include "llvm/ADT/ArrayRef.h"
#include "glow/Base/Tensor.h"



#include <fstream>

using namespace glow;





template <class T>
void fwdMaxPool(Tensor *inW, Tensor *outW,
                llvm::ArrayRef<unsigned_t> kernelSizes,
                llvm::ArrayRef<unsigned_t> strides,
                llvm::ArrayRef<unsigned_t> pads) {
  ShapeNHWC odim(outW->dims());
  ShapeNHWC idim(inW->dims());
  Handle<T> inHandle = inW->getHandle<T>();
  Handle<T> outHandle = outW->getHandle<T>();
  PaddingTLBR pdim(pads);
  ShapeHW kdim(kernelSizes);
  ShapeHW sdim(strides);

  // For each input in the batch:
  for (dim_t n = 0; n < odim.n; n++) {

    // For each layer in the output tensor:
    for (dim_t z = 0; z < idim.c; z++) {
      // For each convolution 'jump' in the input tensor:
      sdim_t x = -sdim_t(pdim.top);
      for (dim_t ax = 0; ax < odim.h; x += sdim.height, ax++) {
        sdim_t y = -sdim_t(pdim.left);
        for (dim_t ay = 0; ay < odim.w; y += sdim.width, ay++) {

          bool first = true;
          T max_value = 0;
          dim_t argmaxNHWC = 0;

          for (dim_t fx = 0; fx < kdim.height; fx++) {
            for (dim_t fy = 0; fy < kdim.width; fy++) {
              sdim_t ox = x + fx;
              sdim_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(idim.h) ||
                  oy >= ssize_t(idim.w)) {
                continue;
              }

              T val = inHandle.at({n, (dim_t)ox, (dim_t)oy, z});
              if (first || (val >= max_value)) {
                first = false;
                max_value = val;
              }
            }
          }
          outHandle.at({n, ax, ay, z}) = max_value;

       } // W
      }   // H
    }     // C
  }       // N
}

#define RESET_IOCTL _IOWR('X', 101, unsigned long)
void xlnk_reset() {

  int xlnkfd = open("/dev/xlnk", O_RDWR | O_CLOEXEC);
  if (xlnkfd < 0) {
    printf("Reset failed - could not open device: %d\n", xlnkfd);
    return;
  }
  if (ioctl(xlnkfd, RESET_IOCTL, 0) < 0) {
    printf("Reset failed - IOCTL failed: %d\n", errno);
  }
  close(xlnkfd);
}


int main()
{
  xlnk_reset();
  srand((unsigned int)time(0));

  dim_t N = 1;
  dim_t H = 14;
  dim_t W = 14;
  dim_t C = 16;
  unsigned_t KH = 2;
  unsigned_t KW = 2;
  unsigned_t stride = 2;
  unsigned_t padding = 1;

  dim_t OH = (H+2*padding-KH)/stride + 1;
  dim_t OW = (W+2*padding-KW)/stride + 1;

  long size = H*W*C;
  int8_t* input = new int8_t[size];
  std::ifstream fis("input.bin", std::ios::in | std::ios::binary);
  if((fis.is_open()) && (fis.seekg(0, std::ios::end), fis.tellg()==size)){
    fis.seekg(0, std::ios::beg);
    int8_t* input = new int8_t[size];
    fis.read((char *)input, size);
    fis.close();
  }
  else{
    for (dim_t i = 0; i < size; i++) {
      input[i] = (int8_t) (rand() % 100);
    }
  }





  std::array<unsigned_t,4> padsS = {padding, padding, padding, padding};
  llvm::ArrayRef<unsigned_t> padShape(padsS);

  std::array<unsigned_t,2> kernelS = {KH, KW};
  llvm::ArrayRef<unsigned_t> kernelShape(kernelS);

  std::array<unsigned_t,2> strideS = {stride, stride};
  llvm::ArrayRef<unsigned_t> strideShape(strideS);


  llvm::SmallVector<dim_t, 3> inputDims = {N, H, W, C};
  Type inputType(ElemKind::Int8QTy, inputDims, 1.0, 0);
  Tensor inputs((void*)input, &inputType);

  std::array<dim_t, 4> outputS{{1, 8, 8, 16}};
  llvm::ArrayRef<dim_t> outputShape(outputS);
  Tensor outputs(ElemKind::Int8QTy, outputShape, 1, 0);
  fwdMaxPool<int8_t>(&inputs, &outputs, kernelShape, strideShape, padShape);

  inputs.toBin("input_back");
  outputs.toBin("output");

  std::cout << "TEST Complete\n";
  return 0;

}