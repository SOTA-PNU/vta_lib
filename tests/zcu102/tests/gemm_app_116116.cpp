#include <iostream>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include "vta/runtime.h"

#include <time.h>


#define RESET_IOCTL _IOWR('X', 101, unsigned long)

using namespace std;

//#include "xtensor/xarray.hpp"
//#include "xtensor/xio.hpp"

#define BITSTREAM_PATH "1x16_i8w8a32_15_15_18_17.bin"

#define BS_FPGA_MAN "/sys/class/fpga_manager/fpga0/firmware"
#define BS_FPGA_MAN_FLAGS "/sys/class/fpga_manager/fpga0/flags"


/*
 * CASE II: A[1][16][1][16] * B[16][16][16][16]
 * 검증용 TEST set. B=1이면 gemm결과는 16
 */
#define INPUT_CH 16
#define OUTPUT_CH  16
// 검증용 TEST set. B=1이면 gemm결과는 108
int8_t A_TEST[1][INPUT_CH][1][16] =
  {{
    {{ -84,    5,   33, -114,   53 ,  89,   39,  -38, -109, -117, -109,   74,  122, -66,   10,    9}},
    {{-19,   33,  -50,  -53,   -9,   79,  -74,  -18,   55,   48,  -65,  -42, -103,  21,  -64 , -79}},
    {{-10, -110 ,   2,   89,   56,   90,  108,  102,  -98,  -50, -127, -110,  -21,  68,  -13, 16}},
    {{  -8 , -54,   29, -103,   -7, -100,   77 , -60,   25,  127,  -62,   85, -113,  77,  -20,  -31}},
    {{ -82,  -92,  -86,   80,   32,  -74,   95,   71,   18, -102,  -32 , -86,  -79, -115  , 36 ,  59}},
    {{-106,  -61,  112,  -27,  -20,   -6,  -15,   78,  -70,  -79,   15,   15,   77, -42,   48, 120}},
    {{-119,    0,  -76,  -26,   48,  113,  -70,   93,   10, -119,  -28,  115,  -70, -96,  -64,  -49}},
    {{  -1,  -43,   35,  -45,  -68,   98,  -96, -128,   50,  -88,   41,   33,    2,     -46,  -52,   39}},
    {{ -16,   53,  -45,   14,  -63,  -74,   -1, -115,  -22,   15,   71,  100,  104,    -107,   53,  126}},
    {{  51,   27,   -3,   38, -122,  -20, -110,   -2,   -5,   -7,  -65,  -22,   93,     -62,  -40,   80}},
    {{  54,  -58,  -77,  -33, -125,   29,   26,   95,  120,  105,   26,   57,  -49,     -17,  122,   48}},
    {{  49,  -11,   36,   21,  -45,   52, -111,   10,   56,  -51,  -33,   87,   97,      41,  106,   81}},
    {{ -59,  116,    9,  -95,  107,   35,  -36, -117,   14,  -22,   73,  123, -128,      16,  127,  -22}},
    {{ 118,   -5,  123, -109,   91,  120,   24,  -90,   29,  -87,   95,  -73, -103,       1,  -46,  -34}},
    {{ 125,   47,   23,  -73,  -74,   18,   64,  104, -104,   83,    9,   66,  -69,     -91,  -94,   98}},
    {{ 116,  -72,  -88,   95,   -7,  -56, -119,   -5,   48,   27, -125,    6,  118,      39,  -35,   53}}
  }};

int8_t B[OUTPUT_CH][INPUT_CH][16][16];
int32_t C[1][OUTPUT_CH][1][16];
int8_t C_FPGA[1][OUTPUT_CH][1][16];  

void set_buf(int8_t* b,int m,int n)
{
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      for(int k=0;k<m;k++) {
        for(int l=0;l<n;l++) {
          b[(i*n*m*n)+(j*m*n)+(k*n)+l] = 1; //(rand()%256)-128; //(i==k ? 1 : 0); //
        }
      }
    }
  }
}

void set_acc_buf(int8_t* b,int m,int n)
{
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      for(int k=0;k<m;k++) {
        for(int l=0;l<n;l++) {
          b[(i*n*m*n)+(j*m*n)+(k*n)+l] = 1; //(rand()%256)-128
        }
      }
    }
  }
}

void gemm_cpu_buf(int8_t* a,int8_t*b, int32_t*c)
{
  for(int co=0;co<16;co++) {
    for(int ci=0;ci<16;ci++) {
      c[(co*16) + ci] = 0;
      for(int ko=0;ko<16;ko++) {
        for(int ki=0;ki<16;ki++) {
          int av = static_cast<int>(a[ko*16+ki]);
          int bv = static_cast<int>(b[co*4096 + ko*256 + ci*16 + ki]);
          c[(co*16)+ci] = c[co*16+ci] + av*bv;
        }
      }
    }
  }  
}


template <typename T>
void print_buf(T* b,int m,int n)
{
  int t;
  cout<<"\n[";
  for(int i=0;i<m;i++) {
      cout<<"\n [";
    for(int j=0;j<n;j++) {
        cout<<"\n   [";
      for(int k=0;k<m;k++) {
          cout<<"\n    [";
        for(int l=0;l<n;l++) {
          
          if(l!=0) cout << ",";

          t=static_cast<T>(b[(i*n*m*i)+(j*m*i)+(k*i)+l]);
          printf("%d",t);
          //cout << t;
        }
        cout<<"]";
      }
      cout<<"\n   ]";
    }
      cout<<"\n  ]";

  }  
  cout<<"\n]\n\n";
}



// from pynqlib.c
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

void writeBitStream()
{
  fstream fs_flag;
  fs_flag.open(BS_FPGA_MAN_FLAGS,std::fstream::out);
  fs_flag << "1";  //  // flag 1 = download bitstream fully

  fstream fs_bitstream;    
  fs_bitstream.open(BS_FPGA_MAN,std::fstream::out);
  fs_bitstream << BITSTREAM_PATH;
}



int finit1(void*)
{
  VTAUopLoopBegin(16,1,0,0);
  VTAUopPush(0,1,0,0,0,0,0,0);
  VTAUopLoopEnd();  
  return 0;
}

int finit2(void*)
{
  VTAUopLoopBegin(16,1,0,1);
  VTAUopPush(0,0,0,0,0,0,0,0);
  VTAUopLoopEnd();
  return 0;
}

int main()
{
  clock_t start, end;
  double result;

  // init seed
  srand((unsigned int)time(0));
  
  // 안해도 뒤에서 open해서 메모리는 mapping시키는데 free안된 경우를 위해 강제 reset
  xlnk_reset();
  writeBitStream();

  // init input data

  // SET input A  
  //set_buf((int8_t*)A,1,16); 

  // SET acc B
  set_acc_buf((int8_t*)B,OUTPUT_CH,INPUT_CH);

  //sram할당
  void* A_buf = VTABufferAlloc( INPUT_CH*16);
  void* B_buf = VTABufferAlloc( INPUT_CH*OUTPUT_CH*16*16);
  void* C_buf = VTABufferAlloc( OUTPUT_CH*16);

  //printf("A:%p, A_Buf:%p, sizeof(A):%d\n", A, A_buf, sizeof(A));
  //printf("B:%p, B_Buf:%p, sizeof(B):%d\n", B, B_buf, sizeof(B));
  
  // DMA copy (DRAM ==> SRAM)
  //VTABufferCopy(A, 0, A_buf, 0, sizeof(A), 1);  
  VTABufferCopy(A_TEST, 0, A_buf, 0, sizeof(A_TEST), 1);  
  VTABufferCopy(B, 0, B_buf, 0, sizeof(B), 1); 

  // uop kernel handle
  void* uopHandle1[2];
  void* uopHandle2[2];  
  uopHandle1[0]=nullptr;
  uopHandle2[0]=nullptr;

  //VTADeviceHandle vtadev = VTADeviceAlloc();
  
  VTACommandHandle vtaCmdH{nullptr};
  vtaCmdH = VTATLSCommandHandle();
  //printf("VTACommand Handle = %p\n",vtaCmdH);

  // set debug mode to dump insn & u-op
  VTASetDebugMode(vtaCmdH,VTA_DEBUG_DUMP_INSN | VTA_DEBUG_DUMP_UOP);

  //printf("A buf = %p\n", A_buf);
  //printf("B buf= %p\n", B_buf);

  start = clock();

  VTAPushGEMMOp(uopHandle1,&finit1,nullptr,0);

  VTADepPush(vtaCmdH,2,1);
  for(int ko=0;ko<INPUT_CH;ko++) {
  
      VTADepPop(vtaCmdH,2,1);

      VTALoadBuffer2D(vtaCmdH , A_buf, ko, 1, 1, 1, 0,0,0,0,0, 2 ); // load inp #define VTA_MEM_ID_INP 2
      VTALoadBuffer2D(vtaCmdH , B_buf, ko, 16, 1, 16, 0,0,0,0,0, 1 ); // load wgt  #define VTA_MEM_ID_WGT 1
      VTADepPush(vtaCmdH,1,2);
      VTADepPop(vtaCmdH,1,2);

      VTAPushGEMMOp(uopHandle2,&finit2,nullptr,0);

      VTADepPush(vtaCmdH,2,1);
  }
  
  VTADepPush(vtaCmdH,2,3);
  //VTADepPop(vtaCmdH,2,1);

  VTADepPop(vtaCmdH,2,3);

  //printf("C buf = %p\n", C_buf);
  VTAStoreBuffer2D(vtaCmdH , 0, 4, C_buf, 0, 16, 1, 16 ); // memory_type 4 = VTA_MEM_ID_OUT

  // 실제 수행
  VTASynchronize(vtaCmdH,1<<31);

  end = clock();
  result = (double)(end - start);
  printf("VTA: GEMM execution time: %.5f ms \n", result);

  printf("\n\nC_FPGA:%p, C_Buf:%p, sizeof(C_FPGA):%d\n", C_FPGA, C_buf, sizeof(C_FPGA));
  VTABufferCopy(C_buf, 0, C_FPGA, 0, sizeof(C_FPGA), 2);  // 1 MemCopyToHost
  print_buf((int8_t*)C_FPGA,1,16);
  printf("-------------\n");

  // free command, queue(inst,uop) memory
  VTARuntimeShutdown();

  VTABufferFree( A_buf );
  VTABufferFree( B_buf );
  VTABufferFree( C_buf );

  return 0;
}
