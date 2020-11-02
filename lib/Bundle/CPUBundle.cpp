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
#include <omp.h>
#include "vta/vtalib/include/Bundle/VTABundle.h"

#include <linux/perf_event.h>
#include <asm/unistd.h>
#define DEBUG_INFO  0

using namespace std;
using namespace glow;


struct PerfEvent {
  perf_event_attr pe;
  int fd = -1;

  PerfEvent(std::uint32_t type, std::uint64_t config) {
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.type = type;
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd == -1) {
      fprintf(stderr, "perf_event_open failed for config 0x%lx\n", config);
      abort();
    }
  }

  void Start() {
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  }

  std::int64_t Stop() {
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    std::int64_t count = 0;
    read(fd, &count, sizeof(count));
    return count;
  }

  ~PerfEvent() { close(fd); }
};

struct ArmPmuEvent : PerfEvent {
  static constexpr std::uint16_t L1I_CACHE_REFILL = 0x01;
  static constexpr std::uint16_t L1I_TLB_REFILL = 0x02;
  static constexpr std::uint16_t L1D_CACHE_REFILL = 0x03;
  static constexpr std::uint16_t L1D_CACHE = 0x04;
  static constexpr std::uint16_t L1D_TLB_REFILL = 0x05;
  static constexpr std::uint16_t LD_RETIRED = 0x06;
  static constexpr std::uint16_t ST_RETIRED = 0x07;
  static constexpr std::uint16_t INST_RETIRED = 0x08;
  static constexpr std::uint16_t EXC_TAKEN = 0x09;
  static constexpr std::uint16_t EXC_RETURN = 0x0A;
  static constexpr std::uint16_t CID_WRITE_RETIRED = 0x0B;
  static constexpr std::uint16_t PC_WRITE_RETIRED = 0x0C;
  static constexpr std::uint16_t BR_IMMED_RETIRED = 0x0D;
  static constexpr std::uint16_t BR_RETURN_RETIRED = 0x0E;
  static constexpr std::uint16_t UNALIGNED_LDST_RETIRED = 0x0F;
  static constexpr std::uint16_t BR_MIS_PRED = 0x10;
  static constexpr std::uint16_t CPU_CYCLES = 0x11;
  static constexpr std::uint16_t BR_PRED = 0x12;
  static constexpr std::uint16_t MEM_ACCESS = 0x13;
  static constexpr std::uint16_t L1I_CACHE = 0x14;
  static constexpr std::uint16_t L1D_CACHE_WB = 0x15;
  static constexpr std::uint16_t L2D_CACHE = 0x16;
  static constexpr std::uint16_t L2D_CACHE_REFILL = 0x17;
  static constexpr std::uint16_t L2D_CACHE_WB = 0x18;
  static constexpr std::uint16_t BUS_ACCESS = 0x19;
  static constexpr std::uint16_t MEMORY_ERROR = 0x1A;
  static constexpr std::uint16_t INST_SPEC = 0x1B;
  static constexpr std::uint16_t TTBR_WRITE_RETIRED = 0x1C;
  static constexpr std::uint16_t BUS_CYCLES = 0x1D;
  static constexpr std::uint16_t CHAIN = 0x1E;
  static constexpr std::uint16_t L1D_CACHE_ALLOCATE = 0x1F;
  static constexpr std::uint16_t L2D_CACHE_ALLOCATE = 0x20;
  static constexpr std::uint16_t BR_RETIRED = 0x21;
  static constexpr std::uint16_t BR_MIS_PRED_RETIRED = 0x22;
  static constexpr std::uint16_t STALL_FRONTEND = 0x23;
  static constexpr std::uint16_t STALL_BACKEND = 0x24;
  static constexpr std::uint16_t L1D_TLB = 0x25;
  static constexpr std::uint16_t L1I_TLB = 0x26;
  static constexpr std::uint16_t L2I_CACHE = 0x27;
  static constexpr std::uint16_t L2I_CACHE_REFILL = 0x28;
  static constexpr std::uint16_t L3D_CACHE_ALLOCATE = 0x29;
  static constexpr std::uint16_t L3D_CACHE_REFILL = 0x2A;
  static constexpr std::uint16_t L3D_CACHE = 0x2B;
  static constexpr std::uint16_t L3D_CACHE_WB = 0x2C;
  static constexpr std::uint16_t L2D_TLB_REFILL = 0x2D;
  static constexpr std::uint16_t L2I_TLB_REFILL = 0x2E;
  static constexpr std::uint16_t L2D_TLB = 0x2F;
  static constexpr std::uint16_t L2I_TLB = 0x30;
  static constexpr std::uint16_t LL_CACHE = 0x32;
  static constexpr std::uint16_t LL_CACHE_MISS = 0x33;
  static constexpr std::uint16_t DTLB_WALK = 0x34;
  static constexpr std::uint16_t LL_CACHE_RD = 0x36;
  static constexpr std::uint16_t LL_CACHE_MISS_RD = 0x37;
  static constexpr std::uint16_t L1D_CACHE_RD = 0x40;
  static constexpr std::uint16_t L1D_CACHE_REFILL_RD = 0x42;
  static constexpr std::uint16_t L1D_TLB_REFILL_RD = 0x4C;
  static constexpr std::uint16_t L1D_TLB_RD = 0x4E;
  static constexpr std::uint16_t L2D_CACHE_RD = 0x50;
  static constexpr std::uint16_t L2D_CACHE_REFILL_RD = 0x52;
  static constexpr std::uint16_t BUS_ACCESS_RD = 0x60;
  static constexpr std::uint16_t MEM_ACCESS_RD = 0x66;
  static constexpr std::uint16_t L3D_CACHE_RD = 0xA0;
  static constexpr std::uint16_t L3D_CACHE_REFILL_RD = 0xA2;
  ArmPmuEvent(std::uint16_t number) : PerfEvent(PERF_TYPE_RAW, number) {}
};

struct CacheCounts {
  int st_retired = 0;
  int ld_retired = 0;
  int mem_access = 0;
  int ll_cache = 0;
  int ll_cache_miss = 0;
  int l1d_tlb = 0;
  int l1d_tlb_refill = 0;
  int l1d_cache = 0;
  int l1d_cache_refill = 0;
  int l2d_cache = 0;
  int l2d_cache_refill = 0;
  int l3d_cache = 0;
  int l3d_cache_refill = 0;
};

void PrintCacheCounts(const CacheCounts& cache_counts) {
 printf("st_retired = %d\n", cache_counts.st_retired);  
  printf("ld_retired = %d\n", cache_counts.ld_retired);
  printf("mem_access = %d\n", cache_counts.mem_access);
  printf("ll_cache = %d\n", cache_counts.ll_cache);
  printf("ll_cache_miss = %d\n", cache_counts.ll_cache_miss);
  printf("l1d_tlb = %d\n", cache_counts.l1d_tlb);
  printf("l1d_tlb_refill = %d\n", cache_counts.l1d_tlb_refill);  
  printf("l1d_cache = %d\n", cache_counts.l1d_cache);
  printf("l1d_cache_refill = %d\n", cache_counts.l1d_cache_refill);
  printf("l2d_cache = %d\n", cache_counts.l2d_cache);
  printf("l2d_cache_refill = %d\n", cache_counts.l2d_cache_refill);
  printf("l3d_cache = %d\n", cache_counts.l3d_cache);
  printf("l3d_cache_refill = %d\n", cache_counts.l3d_cache_refill);
}

const bool only_reads = false;
ArmPmuEvent st_retired(ArmPmuEvent::ST_RETIRED);
ArmPmuEvent ld_retired(ArmPmuEvent::LD_RETIRED);
ArmPmuEvent mem_access(only_reads ? ArmPmuEvent::MEM_ACCESS_RD
                                  : ArmPmuEvent::MEM_ACCESS);
ArmPmuEvent ll_cache(only_reads ? ArmPmuEvent::LL_CACHE_RD
                                : ArmPmuEvent::LL_CACHE);
ArmPmuEvent ll_cache_miss(only_reads ? ArmPmuEvent::LL_CACHE_MISS_RD
                                      : ArmPmuEvent::LL_CACHE_MISS);
ArmPmuEvent l1d_cache(only_reads ? ArmPmuEvent::L1D_CACHE_RD
                                  : ArmPmuEvent::L1D_CACHE);
ArmPmuEvent l1d_cache_refill(only_reads ? ArmPmuEvent::L1D_CACHE_REFILL_RD
                                        : ArmPmuEvent::L1D_CACHE_REFILL);
ArmPmuEvent l1d_tlb(only_reads ? ArmPmuEvent::L1D_TLB_RD
                                  : ArmPmuEvent::L1D_TLB);
ArmPmuEvent l1d_tlb_refill(only_reads ? ArmPmuEvent::L1D_TLB_REFILL_RD
                                        : ArmPmuEvent::L1D_TLB_REFILL);                                        
ArmPmuEvent l2d_cache(only_reads ? ArmPmuEvent::L2D_CACHE_RD
                                  : ArmPmuEvent::L2D_CACHE);
ArmPmuEvent l2d_cache_refill(only_reads ? ArmPmuEvent::L2D_CACHE_REFILL_RD
                                        : ArmPmuEvent::L2D_CACHE_REFILL);
ArmPmuEvent l3d_cache(only_reads ? ArmPmuEvent::L3D_CACHE_RD
                                  : ArmPmuEvent::L3D_CACHE);
ArmPmuEvent l3d_cache_refill(only_reads ? ArmPmuEvent::L3D_CACHE_REFILL_RD
                                        : ArmPmuEvent::L3D_CACHE_REFILL);

void StartCacheCounts() {
  st_retired.Start();
  ld_retired.Start();
  mem_access.Start();
  ll_cache.Start();
  ll_cache_miss.Start();
  l1d_cache.Start();
  l1d_cache_refill.Start();
  l1d_tlb.Start();
  l1d_tlb_refill.Start();

  l2d_cache.Start();
  l2d_cache_refill.Start();
  l3d_cache.Start();
  l3d_cache_refill.Start();
}

void StopCacheCounts(CacheCounts* cache_counts) 
{
  cache_counts->st_retired = st_retired.Stop();
  cache_counts->ld_retired = ld_retired.Stop();
  cache_counts->mem_access = mem_access.Stop();
  cache_counts->ll_cache = ll_cache.Stop();
  cache_counts->ll_cache_miss = ll_cache_miss.Stop();
  cache_counts->l1d_cache = l1d_cache.Stop();
  cache_counts->l1d_cache_refill = l1d_cache_refill.Stop();
  cache_counts->l1d_tlb = l1d_tlb.Stop();
  cache_counts->l1d_tlb_refill = l1d_tlb_refill.Stop();  
  cache_counts->l2d_cache = l2d_cache.Stop();
  cache_counts->l2d_cache_refill = l2d_cache_refill.Stop();
  cache_counts->l3d_cache = l3d_cache.Stop();
  cache_counts->l3d_cache_refill = l3d_cache_refill.Stop();
}


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

// (16xK)*(K*8n). width_a = K = depth,  width_b =  * elemsize(1). 
// best performance if K = 16k
// lhs: row-major, rhs: colomn-major
// 4*(KW*KH) x (KW*KH)*(out_w*out_h)
void mm_matrix_multiply_s8_v2(int8_t *lhs_ptr, int8_t *rhs_ptr, int32_t *accum_ptr, int width_a, int width_b, size_t stride_b, size_t stride_c, int32_t filterOffset)
{
  
  // block = 1x16 * 16*8
  
  //TODO. temp
  // check overflow 
  // upscale to s32 if s16 datatype overflow because of filterOffset  
  assert(filterOffset < 32700 );
  assert(filterOffset > -32700);

  const int16x4_t vfilterOff = vdup_n_s16(filterOffset);





  for (int nblock = 0; nblock < width_b / 4; nblock++)
  {
   
      int8_t *a_ptr = lhs_ptr;                         
      int8_t *b_ptr = rhs_ptr + nblock * 4 * stride_b; 

      int left = width_a % 16;

    int32x4_t acc[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            acc[i][j] = vdupq_n_s32(0);
        }
    }
    // 16개 단위만 먼저 처리
    for (int d = 0; d < width_a - left ; d += 16)
    {
        int8x16_t lhs[4];
        for (int i = 0; i < 4; i++)
        {
            lhs[i] = vld1q_s8(a_ptr + width_a * i);
        }
        int8x16_t rhs[4];
        for (int i = 0; i < 4; i++)
        {
            rhs[i] = vld1q_s8(b_ptr + stride_b * i);
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
        a_ptr += 16;  
        b_ptr += 16;
    }

    // 16보다 작은거 처리
    {
        int8x16_t lhs[4];
        for (int i = 0; i < 4; i++)
        {
            lhs[i] = vld1q_s8(a_ptr + width_a * i);
        }
        int8x16_t rhs[4];
        for (int i = 0; i < 4; i++)
        {
            rhs[i] = vld1q_s8(b_ptr + stride_b * i);
        }

        int idx=0;
        if(left >= 8) {
          for (int i = 0; i < 4; i++)
          {
              for (int j = 0; j < 4; j++)
              {
                  int16x8_t local_acc =
                      vmull_s8(vget_low_s8(lhs[i]), vget_low_s8(rhs[j]));
                  //local_acc =
                  //    vmlal_s8(local_acc, vget_high_s8(lhs[i]), vget_high_s8(rhs[j]));
                  acc[i][j] = vpadalq_s16(acc[i][j], local_acc);
              }
          }
          left -= 8;
          idx+=8;
        }

        if(left>=4) {
          for (int i = 0; i < 4; i++)
          {
              for (int j = 0; j < 4; j++)
              {
                int16x8_t local_acc;
                if(idx==0) {
                            local_acc =
                            vmull_s8(vget_low_s8(lhs[i]), vget_low_s8(rhs[j]));
                }
                else if(idx==8) {          
                        local_acc =
                            vmull_s8(vget_high_s8(lhs[i]), vget_high_s8(rhs[j]));
                }

                  acc[i][j] = vaddq_s32(acc[i][j], vmovl_s16(vget_low_s16(local_acc)));
              }
          }

          idx+=4;
          left-=4;
        }

    }
    
    for (int i = 0; i < 4; i++)
    {
        int32x4_t acc_2x_0 = vpaddq_s32(acc[0][i], acc[1][i]);
        int32x4_t acc_2x_1 = vpaddq_s32(acc[2][i], acc[3][i]);
        int32x4_t acc_4x = vpaddq_s32(acc_2x_0, acc_2x_1);
        int32x4_t dst_val = vld1q_s32(accum_ptr + 4 * i);
        dst_val = vaddq_s32(dst_val, acc_4x);
        vst1q_s32(accum_ptr +  nblock * 4 + stride_c * i, dst_val);
    }

  } // n_block  
   
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
  assert(filterOffset > -32700);

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

// (16xK)*(K*8n). width_a = K = depth,  width_b =  * elemsize(1). 
// best performance if K = 16k
// lhs: row-major, rhs: colomn-major
void mm_matrix_multiply_s8(int8_t *lhs_ptr, int8_t *rhs_ptr, int32_t *accum_ptr, int width_a, int width_b, size_t stride_b, int32_t filterOffset)
{
  
  // block = 1x16 * 16*8
  
  //TODO. temp
  // check overflow 
  // upscale to s32 if s16 datatype overflow because of filterOffset  
  assert(filterOffset < 32700 );
  assert(filterOffset > -32700);

  const int16x4_t vfilterOff = vdup_n_s16(filterOffset);
  const int k_row=8;  // 커널 갯수
  // lhs = (k_row) * 16
  // rhs = 16*8 (8개 width)

  for (int nblock = 0; nblock < width_b / 8; nblock++)
  {
      // 8 width
      int32x4_t c[k_row][8];

      int8x16_t a_s8[k_row] ;
      int16x4x4_t a_s16[k_row];

      for(int k =0;k<k_row;k++)
      for (int o = 0; o < 8; o++)
          c[k][o] = vdupq_n_s32(0);

      int8_t *a_ptr = lhs_ptr;                         
      int8_t *b_ptr = rhs_ptr + nblock * 8 * stride_b; 
      auto vec_a_end_addr = a_ptr + width_a;

      if( vec_a_end_addr - a_ptr >= 16)
      {
        // This for loop performs 8 accumulations
        // cell layout: 1x16 * 16*8 = 1x8
        for (; a_ptr <= (vec_a_end_addr - 16);)
        {

            
          /*  const int8x16_t a40_s8 = vld1q_s8(a_ptr + 4* width_a);
            const int8x16_t a50_s8 = vld1q_s8(a_ptr + 5* width_a);
            const int8x16_t a60_s8 = vld1q_s8(a_ptr + 6* width_a);
            const int8x16_t a70_s8 = vld1q_s8(a_ptr + 7* width_a);            
*/
            const int8x16_t b00_s8 = vld1q_s8(b_ptr + 0 * stride_b);
            const int8x16_t b10_s8 = vld1q_s8(b_ptr + 1 * stride_b);
            const int8x16_t b20_s8 = vld1q_s8(b_ptr + 2 * stride_b);
            const int8x16_t b30_s8 = vld1q_s8(b_ptr + 3 * stride_b);
            const int8x16_t b40_s8 = vld1q_s8(b_ptr + 4 * stride_b);
            const int8x16_t b50_s8 = vld1q_s8(b_ptr + 5 * stride_b);
            const int8x16_t b60_s8 = vld1q_s8(b_ptr + 6 * stride_b);
            const int8x16_t b70_s8 = vld1q_s8(b_ptr + 7 * stride_b);

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

            for(int k=0;k<k_row;k++)
            {
                 a_s8[k] = vld1q_s8(a_ptr + k * width_a);
                a_s16[k] =
                {
                    {vsub_s16(vget_low_s16(vmovl_s8(vget_low_s8(a_s8[k]))), vfilterOff),
                    vsub_s16(vget_high_s16(vmovl_s8(vget_low_s8(a_s8[k]))), vfilterOff),
                    vsub_s16(vget_low_s16(vmovl_s8(vget_high_s8(a_s8[k]))), vfilterOff),
                    vsub_s16(vget_high_s16(vmovl_s8(vget_high_s8(a_s8[k]))), vfilterOff)}};

              // Accumulate 0:
              c[k][0] = vmlal_s16(c[k][0], b00_s16.val[0], a_s16[k].val[0]);
              c[k][0] = vmlal_s16(c[k][0], b00_s16.val[1], a_s16[k].val[1]);
              c[k][0] = vmlal_s16(c[k][0], b00_s16.val[2], a_s16[k].val[2]);
              c[k][0] = vmlal_s16(c[k][0], b00_s16.val[3], a_s16[k].val[3]);

              // Accumulate 1:
              c[k][1] = vmlal_s16(c[k][1], b10_s16.val[0], a_s16[k].val[0]);
              c[k][1] = vmlal_s16(c[k][1], b10_s16.val[1], a_s16[k].val[1]);
              c[k][1] = vmlal_s16(c[k][1], b10_s16.val[2], a_s16[k].val[2]);
              c[k][1] = vmlal_s16(c[k][1], b10_s16.val[3], a_s16[k].val[3]);

              // Accumulate 2:
              c[k][2] = vmlal_s16(c[k][2], b20_s16.val[0], a_s16[k].val[0]);
              c[k][2] = vmlal_s16(c[k][2], b20_s16.val[1], a_s16[k].val[1]);
              c[k][2] = vmlal_s16(c[k][2], b20_s16.val[2], a_s16[k].val[2]);
              c[k][2] = vmlal_s16(c[k][2], b20_s16.val[3], a_s16[k].val[3]);

              // Accumulate 3:
              c[k][3] = vmlal_s16(c[k][3], b30_s16.val[0], a_s16[k].val[0]);
              c[k][3] = vmlal_s16(c[k][3], b30_s16.val[1], a_s16[k].val[1]);
              c[k][3] = vmlal_s16(c[k][3], b30_s16.val[2], a_s16[k].val[2]);
              c[k][3] = vmlal_s16(c[k][3], b30_s16.val[3], a_s16[k].val[3]);

              // Accumulate 4:
              c[k][4] = vmlal_s16(c[k][4], b40_s16.val[0], a_s16[k].val[0]);
              c[k][4] = vmlal_s16(c[k][4], b40_s16.val[1], a_s16[k].val[1]);
              c[k][4] = vmlal_s16(c[k][4], b40_s16.val[2], a_s16[k].val[2]);
              c[k][4] = vmlal_s16(c[k][4], b40_s16.val[3], a_s16[k].val[3]);

              // Accumulate 5:
              c[k][5] = vmlal_s16(c[k][5], b50_s16.val[0], a_s16[k].val[0]);
              c[k][5] = vmlal_s16(c[k][5], b50_s16.val[1], a_s16[k].val[1]);
              c[k][5] = vmlal_s16(c[k][5], b50_s16.val[2], a_s16[k].val[2]);
              c[k][5] = vmlal_s16(c[k][5], b50_s16.val[3], a_s16[k].val[3]);

              // Accumulate 6:
              c[k][6] = vmlal_s16(c[k][6], b60_s16.val[0], a_s16[k].val[0]);
              c[k][6] = vmlal_s16(c[k][6], b60_s16.val[1], a_s16[k].val[1]);
              c[k][6] = vmlal_s16(c[k][6], b60_s16.val[2], a_s16[k].val[2]);
              c[k][6] = vmlal_s16(c[k][6], b60_s16.val[3], a_s16[k].val[3]);

              // Accumulate 7:
              c[k][7] = vmlal_s16(c[k][7], b70_s16.val[0], a_s16[k].val[0]);
              c[k][7] = vmlal_s16(c[k][7], b70_s16.val[1], a_s16[k].val[1]);
              c[k][7] = vmlal_s16(c[k][7], b70_s16.val[2], a_s16[k].val[2]);
              c[k][7] = vmlal_s16(c[k][7], b70_s16.val[3], a_s16[k].val[3]);
            }

            a_ptr += 16;
            b_ptr += 16;

#if DEBUG_INFO
  if(nblock==0) {
    
    int32_t sum_temp[4];
    vst1q_s32(sum_temp, c0[k][0]);

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
         

          //const int8x16_t a00_s8 = vld1q_s8(a_ptr);
          const int8x16_t b00_s8 = vld1q_s8(b_ptr + 0 * stride_b);
          const int8x16_t b10_s8 = vld1q_s8(b_ptr + 1 * stride_b);
          const int8x16_t b20_s8 = vld1q_s8(b_ptr + 2 * stride_b);
          const int8x16_t b30_s8 = vld1q_s8(b_ptr + 3 * stride_b);
          const int8x16_t b40_s8 = vld1q_s8(b_ptr + 4 * stride_b);
          const int8x16_t b50_s8 = vld1q_s8(b_ptr + 5 * stride_b);
          const int8x16_t b60_s8 = vld1q_s8(b_ptr + 6 * stride_b);
          const int8x16_t b70_s8 = vld1q_s8(b_ptr + 7 * stride_b);


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

          // Convert a00_s8 to int16_t and get the lower part
            // Convert a00_s8 to int16_t and get the lower part
            for(int k=0;k<k_row;k++)
            {
                a_s8[k] = vld1q_s8(a_ptr + k * width_a);
                a_s16[k] =
                {
                    {vsub_s16(vget_low_s16(vmovl_s8(vget_low_s8(a_s8[k]))), vfilterOff),
                    vsub_s16(vget_high_s16(vmovl_s8(vget_low_s8(a_s8[k]))), vfilterOff),
                    vsub_s16(vget_low_s16(vmovl_s8(vget_high_s8(a_s8[k]))), vfilterOff),
                    vsub_s16(vget_high_s16(vmovl_s8(vget_high_s8(a_s8[k]))), vfilterOff)}};

            }

            int ii = 0; // index 
            if(vec_a_end_addr - a_ptr  >= 8) {
         
              for(int k=0;k<k_row;k++)
              {
                   // Accumulate 0:
                c[k][0] = vmlal_s16(c[k][0], b00_s16.val[0+ii], a_s16[k].val[0+ii]);
                c[k][0] = vmlal_s16(c[k][0], b00_s16.val[1+ii], a_s16[k].val[1+ii]);

                // Accumulate 1:
                c[k][1] = vmlal_s16(c[k][1], b10_s16.val[0+ii], a_s16[k].val[0+ii]);
                c[k][1] = vmlal_s16(c[k][1], b10_s16.val[1+ii], a_s16[k].val[1+ii]);

                // Accumulate 2:
                c[k][2] = vmlal_s16(c[k][2], b20_s16.val[0+ii], a_s16[k].val[0+ii]);
                c[k][2] = vmlal_s16(c[k][2], b20_s16.val[1+ii], a_s16[k].val[1+ii]);

                // Accumulate 3:
                c[k][3] = vmlal_s16(c[k][3], b30_s16.val[0+ii], a_s16[k].val[0+ii]);
                c[k][3] = vmlal_s16(c[k][3], b30_s16.val[1+ii], a_s16[k].val[1+ii]);

                // Accumulate 4:
                c[k][4] = vmlal_s16(c[k][4], b40_s16.val[0+ii], a_s16[k].val[0+ii]);
                c[k][4] = vmlal_s16(c[k][4], b40_s16.val[1+ii], a_s16[k].val[1+ii]);

                // Accumulate 5:
                c[k][5] = vmlal_s16(c[k][5], b50_s16.val[0+ii], a_s16[k].val[0+ii]);
                c[k][5] = vmlal_s16(c[k][5], b50_s16.val[1+ii], a_s16[k].val[1+ii]);

                // Accumulate 6:
                c[k][6] = vmlal_s16(c[k][6], b60_s16.val[0+ii], a_s16[k].val[0+ii]);
                c[k][6] = vmlal_s16(c[k][6], b60_s16.val[1+ii], a_s16[k].val[1+ii]);

                // Accumulate 7:
                c[k][7] = vmlal_s16(c[k][7], b70_s16.val[0+ii], a_s16[k].val[0+ii]);
                c[k][7] = vmlal_s16(c[k][7], b70_s16.val[1+ii], a_s16[k].val[1+ii]);
              }
                a_ptr+=8;
                ii+= (8/4);
              
            }


            if(vec_a_end_addr - a_ptr >= 4) {
                      for(int k=0;k<k_row;k++)
              {
              // Accumulate 0:
              c[k][0] = vmlal_s16(c[k][0], b00_s16.val[0+ii], a_s16[k].val[0+ii]);

              // Accumulate 1:
              c[k][1] = vmlal_s16(c[k][1], b10_s16.val[0+ii], a_s16[k].val[0+ii]);

              // Accumulate 2:
              c[k][2] = vmlal_s16(c[k][2], b20_s16.val[0+ii], a_s16[k].val[0+ii]);

              // Accumulate 3:
              c[k][3] = vmlal_s16(c[k][3], b30_s16.val[0+ii], a_s16[k].val[0+ii]);

              // Accumulate 4:
              c[k][4] = vmlal_s16(c[k][4], b40_s16.val[0+ii], a_s16[k].val[0+ii]);

              // Accumulate 5:
              c[k][5] = vmlal_s16(c[k][5], b50_s16.val[0+ii], a_s16[k].val[0+ii]);

              // Accumulate 6:
              c[k][6] = vmlal_s16(c[k][6], b60_s16.val[0+ii], a_s16[k].val[0+ii]);

              // Accumulate 7:
              c[k][7] = vmlal_s16(c[k][7], b70_s16.val[0+ii], a_s16[k].val[0+ii]);
              }

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
        for(int k=0;k<k_row;k++)
              {
            // Accumulate 0:
            c[k][0] = vmlal_s16(c[k][0], b00_s16.val[0+ii], vmul_s16(left_mask, a_s16[k].val[0+ii]));

            // Accumulate 1:
            c[k][1] = vmlal_s16(c[k][1], b10_s16.val[0+ii], vmul_s16(left_mask, a_s16[k].val[0+ii]));

            // Accumulate 2:
            c[k][2] = vmlal_s16(c[k][2], b20_s16.val[0+ii], vmul_s16(left_mask, a_s16[k].val[0+ii]));

            // Accumulate 3:
            c[k][3] = vmlal_s16(c[k][3], b30_s16.val[0+ii], vmul_s16(left_mask, a_s16[k].val[0+ii]));

            // Accumulate 4:
            c[k][4] = vmlal_s16(c[k][4], b40_s16.val[0+ii], vmul_s16(left_mask, a_s16[k].val[0+ii]));

            // Accumulate 5:
            c[k][5] = vmlal_s16(c[k][5], b50_s16.val[0+ii], vmul_s16(left_mask, a_s16[k].val[0+ii]));

            // Accumulate 6:
            c[k][6] = vmlal_s16(c[k][6], b60_s16.val[0+ii], vmul_s16(left_mask, a_s16[k].val[0+ii]));

            // Accumulate 7:
            c[k][7] = vmlal_s16(c[k][7], b70_s16.val[0+ii], vmul_s16(left_mask, a_s16[k].val[0+ii]));

              }
            a_ptr += left;
          }
          // b_ptr += stride_b;
      }


      for (int k = 0; k < k_row; k++)
      {

        for (int o = 0; o < 8; o++)
        {
            int32x2_t temp = vadd_s32(vget_high_s32(c[k][o]), vget_low_s32(c[k][o]));
            int32x2_t temp2 = vpadd_s32(temp, temp);
            int32_t sum1 = vget_lane_s32(temp2, 0);

            //vst1q_s32(accum_ptr + nblock * 8 + k, c0[k]);
            accum_ptr[k*width_b + nblock * 8  + o] = sum1;
      
        }
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

// nhwc input matrix to col
/*
void im2col_nhwc(int8_t* in, int8_t* out, int inw, int inh, int inch, int outw, int outh, int kernel_h, int kernel_w, int stride_h, int stride_w )
{
    const int stride = kernel_h * kernel_w * outw * outh;
    
    for (int p = 0; p < inch; p++)
    {
        const signed char* input_ch_p = in + (p* inw*inh );
        int retID = stride * p;
        for (int u = 0; u < kernel_h; u++)
        {
            for (int v = 0; v < kernel_w; v++)
            {
                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        int row = u + i * stride_h;
                        int col = v + j * stride_w;
                        int index = row * inw + col;
                        out[retID] = input_ch_p[index];
                        retID++;
                    }
                }
            }
        }
    }
}
*/
inline void linearize_volume_nchw(const int8_t *const in_ptr,
                                  int8_t *out_ptr,
                                  int top_left_x,
                                  int top_left_y,
                                  int kernel_width,
                                  int kernel_height,
                                  int kernel_depth,
                                  int input_w,
                                  int input_h,
                                  int input_stride_x,
                                  int input_stride_y,
                                  int input_stride_z,
                                  int has_pads,
                                  int pad_value,
                                  int dilation_x,
                                  int dilation_y,
                                  int32_t input_offset)
{
    const int kernel_size2 = kernel_width * kernel_height;
    const int x_e = top_left_x + kernel_width * dilation_x;
    const int y_e = top_left_y + kernel_height * dilation_y;

    // Linearize volume
    int d = 0;
    // This for loop linearize a volume with 3 slices. This allows:
    // 1) to reduce the iterations of the outer for loop "d"
    // 2) to have an optimized im2col for the first convolution layer where usually we have 3 IFMs
    for (; d <= (kernel_depth - 3); d += 3)
    {
        for (int y = top_left_y; y < y_e; y += dilation_y)
        {
            if ((y < 0 || y >= input_h) && has_pads)
            {
                // All the values will be the offset (will be zeros when not quantized)
                for (int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    *(out_ptr + 0 * kernel_size2) = pad_value;
                    *(out_ptr + 1 * kernel_size2) = pad_value;
                    *(out_ptr + 2 * kernel_size2) = pad_value;
                }
            }
            else
            {
                for (int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    if ((x < 0 || x >= input_w) && has_pads)
                    {
                        *(out_ptr + 0 * kernel_size2) = pad_value;
                        *(out_ptr + 1 * kernel_size2) = pad_value;
                        *(out_ptr + 2 * kernel_size2) = pad_value;
                    }
                    else
                    {
                        *(out_ptr + 0 * kernel_size2) = *((in_ptr + ((d + 0) * input_stride_z + y * input_stride_y + x * input_stride_x))) - input_offset;
                        *(out_ptr + 1 * kernel_size2) = *((in_ptr + ((d + 1) * input_stride_z + y * input_stride_y + x * input_stride_x)))  - input_offset;
                        *(out_ptr + 2 * kernel_size2) = *((in_ptr + ((d + 2) * input_stride_z + y * input_stride_y + x * input_stride_x)))  - input_offset;
                    }
                }
            }
        }
        out_ptr += 2 * kernel_size2;
    }

    // Left over
    for (; d < kernel_depth; d++)
    {
        for (int y = top_left_y; y < y_e; y += dilation_y)
        {
            if ((y < 0 || y >= input_h) && has_pads)
            {
                // All the values will be the offset (will be zeros when not quantized)
                memset(out_ptr, pad_value, kernel_width);
                out_ptr += kernel_width;
            }
            else
            {
                for (int x = top_left_x; x < x_e; x += dilation_x, ++out_ptr)
                {
                    if ((x < 0 || x >= input_w) && has_pads)
                    {
                        *out_ptr = pad_value;
                    }
                    else
                    {
                        *out_ptr = *((in_ptr + (d * input_stride_z + y * input_stride_y + x * input_stride_x))) -  - input_offset;
                    }
                }
            }
        }
    }
}


// input nchw, output nchw
void fwdConvolution_nchw(int8_t *input, int8_t *kernel, int8_t *bias, int8_t *output, 
                      float inS, float kS, float bS, float outS,
                      int32_t inOffset, int32_t filterOffset, int32_t biasOffset, int32_t outOffset,
                      dim_t N, dim_t C, dim_t H, dim_t W, dim_t KN, dim_t KH, dim_t KW, int pad_size, int stride_size, size_t group,
                      size_t dilation, bool doRelu, bool doBias, 
                      int out_h, int out_w) {

  // only support out_w is 4*n
  assert(out_w %4 ==0);
  
  float outScale = outS;
  float inScale = inS;
  float filterScale = kS;
  float biasScale = bS;

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
    float matMulScale = inScale * filterScale;
    float scale = matMulScale / outScale;
    float bscale = biasScale / matMulScale;  

#if DEBUG_INFO
  printf("off i:%d f:%d b:%d o:%d\n",inOffset,  filterOffset,  biasOffset,  outOffset);

  printf("\nscale:%f\n", scale);
  printf("bscale:%f\n",bscale);
#endif


  int8_t* flat_input = (int8_t*)malloc (KH*KW*C* out_h * out_w); // c= channel
  int8_t* kernel_vec = (int8_t*)malloc (KH*KW*C); // 1 filter * channel
  int32_t* mm_out_temp = (int32_t*)malloc(out_h*out_w*sizeof(int32_t)); // output of 1 filter for all channel
  
  
  
  // make patched input
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -ssize_t(pad_size);
  for (dim_t ax = 0; ax < out_h; x += stride_size, ax++) {  //H
    ssize_t y = -ssize_t(pad_size);
    for (dim_t ay = 0; ay < out_w; y += stride_size, ay++) { 
       linearize_volume_nchw(input, flat_input + (ax * KH * KW * out_w * C) + (ay * KH * KW * C), x, y, KW, KH, C, W, H, 1, W, W * H, pad_size > 0 ? 1 : 0, 0, 1, 1, inOffset);
    
#if DEBUG_INFO     
      if(ax>4 && ay <4) 
      {
        for(int c=0;c<3;c++)
        {
        printf("\n(%d,%d - %d)[%d,%d]\n",x,y,c,KW,KH);
        //ch0�� ���. 
        for(int k=0;k<KW*KH;k++) { printf("%02x ",flat_input[k + (ax * KH * KW * out_w * C) + (ay * KH * KW * C)+ c*KH*KW]); if(k%7==6) printf("/ "); }

        }
      }
#endif      
      
    }
  }

  // For each output channel in the group:
  for (dim_t d = 0; d < KN; d++) {

      // load 1 filter * all channel
      memcpy(kernel_vec, kernel + KH * KW * C * d, KH * KW * C);

#if DEBUG_INFO
      printf("kernel %d\n",d);
      for(int c=0;c<C;c++)
      {
        for (int j = 0; j < KH; j++)
        {
          for (int i = 0; i < KW; i++)
            {
                printf("%x ",kernel_vec[i + j * KW + c*KH*KW]);
            }
          printf(" / ");
        }
        printf("\n");
      }
#endif

      vector_matrix_multiply_s8(kernel_vec, flat_input, (int32_t *)mm_out_temp, KH * KW * C, out_h * out_w, KH * KW * C, filterOffset);
      
#if DEBUG_INFO
    if(d<KN) 
    {
        
        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < 2; i++)
            {
                printf("%d ",mm_out_temp[i + j * out_w]+ *((int32_t*)bias+d));

            }
            printf("\n");
        }
        
    }
#endif

    vector_scale_bias_offset_clip_s8(mm_out_temp, *((int32_t*)bias+d), output  + (d*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
  
#if DEBUG_INFO
    if(d<KN) 
    {
        
        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < 2; i++)
            {
                printf("(%d)",*(output  + (d*out_h*out_w) + i + j * out_w) );
                if(i % 16 == 15) printf(" / ");
            }
            printf("\n");
        }
        
    }
#endif    

  }

 
  
}
static void reorder_b(const int8_t* b, int8_t* sb, const int k, const int n, const int ldx)
{
#if PRINT_MATRIX
    print_int8_matrix("b", b, k, n, ldx);
    int8_t* origin = sb;
#endif
    int i = 0;
    for (; i + 3 < n; i += 4)
    {
        const int8_t* p0 = b + i;
        const int8_t* p1 = b + 1 * ldx + i;
        const int8_t* p2 = b + 2 * ldx + i;
        const int8_t* p3 = b + 3 * ldx + i;

        const int8_t* p4 = b + 4 * ldx + i;
        const int8_t* p5 = b + 5 * ldx + i;
        const int8_t* p6 = b + 6 * ldx + i;
        const int8_t* p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j + 7 < k; j += 8)
        {
            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p2[0];
            sb[3] = p3[0];
            sb[4] = p4[0];
            sb[5] = p5[0];
            sb[6] = p6[0];
            sb[7] = p7[0];

            sb[8] = p0[1];
            sb[9] = p1[1];
            sb[10] = p2[1];
            sb[11] = p3[1];
            sb[12] = p4[1];
            sb[13] = p5[1];
            sb[14] = p6[1];
            sb[15] = p7[1];

            sb[16] = p0[2];
            sb[17] = p1[2];
            sb[18] = p2[2];
            sb[19] = p3[2];
            sb[20] = p4[2];
            sb[21] = p5[2];
            sb[22] = p6[2];
            sb[23] = p7[2];

            sb[24] = p0[3];
            sb[25] = p1[3];
            sb[26] = p2[3];
            sb[27] = p3[3];
            sb[28] = p4[3];
            sb[29] = p5[3];
            sb[30] = p6[3];
            sb[31] = p7[3];

            sb += 32;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        if (j + 3 < k)
        {
            j += 4;

            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p2[0];
            sb[3] = p3[0];

            sb[4] = p0[1];
            sb[5] = p1[1];
            sb[6] = p2[1];
            sb[7] = p3[1];

            sb[8] = p0[2];
            sb[9] = p1[2];
            sb[10] = p2[2];
            sb[11] = p3[2];

            sb[12] = p0[3];
            sb[13] = p1[3];
            sb[14] = p2[3];
            sb[15] = p3[3];

            sb += 16;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        if (j + 1 < k)
        {
            j += 2;

            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p0[1];
            sb[3] = p1[1];
            sb[4] = p0[2];
            sb[5] = p1[2];
            sb[6] = p0[3];
            sb[7] = p1[3];

            sb += 8;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        if (j < k)
        {
            sb[0] = p0[0];
            sb[1] = p0[1];
            sb[2] = p0[2];
            sb[3] = p0[3];

            sb += 4;
            p0 += ldx;
        }
    }
    if (i + 1 < n)
    {
        const int8_t* p0 = b + i;
        const int8_t* p1 = b + 1 * ldx + i;
        const int8_t* p2 = b + 2 * ldx + i;
        const int8_t* p3 = b + 3 * ldx + i;

        const int8_t* p4 = b + 4 * ldx + i;
        const int8_t* p5 = b + 5 * ldx + i;
        const int8_t* p6 = b + 6 * ldx + i;
        const int8_t* p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j + 7 < k; j += 8)
        {
            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p2[0];
            sb[3] = p3[0];
            sb[4] = p4[0];
            sb[5] = p5[0];
            sb[6] = p6[0];
            sb[7] = p7[0];

            sb[8] = p0[1];
            sb[9] = p1[1];
            sb[10] = p2[1];
            sb[11] = p3[1];
            sb[12] = p4[1];
            sb[13] = p5[1];
            sb[14] = p6[1];
            sb[15] = p7[1];

            sb += 16;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        if (j + 3 < k)
        {
            j += 4;

            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p2[0];
            sb[3] = p3[0];

            sb[4] = p0[1];
            sb[5] = p1[1];
            sb[6] = p2[1];
            sb[7] = p3[1];

            sb += 8;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        if (j + 1 < k)
        {
            j += 2;

            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p0[1];
            sb[3] = p1[1];

            sb += 4;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        if (j < k)
        {
            sb[0] = p0[0];
            sb[1] = p0[1];

            sb += 2;
            p0 += ldx;
        }
        i += 2;
    }
    if (i < n)
    {
        const int8_t* p0 = b + i;
        const int8_t* p1 = b + 1 * ldx + i;
        const int8_t* p2 = b + 2 * ldx + i;
        const int8_t* p3 = b + 3 * ldx + i;
        const int8_t* p4 = b + 4 * ldx + i;
        const int8_t* p5 = b + 5 * ldx + i;
        const int8_t* p6 = b + 6 * ldx + i;
        const int8_t* p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j + 7 < k; j += 8)
        {
            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p2[0];
            sb[3] = p3[0];
            sb[4] = p4[0];
            sb[5] = p5[0];
            sb[6] = p6[0];
            sb[7] = p7[0];

            sb += 8;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        if (j + 3 < k)
        {
            j += 4;

            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p2[0];
            sb[3] = p3[0];

            sb += 4;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        if (j + 1 < k)
        {
            j += 2;

            sb[0] = p0[0];
            sb[1] = p1[0];

            sb += 2;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        if (j < k)
        {
            sb[0] = p0[0];

            sb += 1;
            p0 += ldx;
        }
    }
#if PRINT_MATRIX
    print_int8_matrix("sb", origin, k, n, n);
#endif
}


#define DECOMPOSE_K                      \
    int ktmp = k;                        \
    int k8 = k >> 3;                     \
    int k8_even = (k8 % 2 == 0) ? 0 : 1; \
    k -= (k8 << 3);                      \
    int k4 = k >> 2;                     \
    k -= (k4 << 2);                      \
    int k2 = k >> 1;                     \
    k -= (k2 << 1);                      \
    int k1 = k;                          \
    k = ktmp;

#define DECOMPOSE_N  \
    int ntmp = n;    \
    int n4 = n >> 2; \
    n -= (n4 << 2);  \
    int n2 = n >> 1; \
    n -= (n2 << 1);  \
    int n1 = n;      \
    n = ntmp;


static void int8kernel_m1(void* dst, int8_t* sa, int8_t* sb, int, int k, int n, int, float* scales, float* bias)
{
    void* pc = dst;
    int8_t* pa = sa;
    int8_t* pb = sb;

    DECOMPOSE_K
    DECOMPOSE_N

    if (n4 > 0)
    {
        asm volatile(
            "9:                               \n"
            "    eor v8.16b, v8.16b, v8.16b   \n"
            "    eor v9.16b, v9.16b, v9.16b   \n"
            "    eor v10.16b, v10.16b, v10.16b\n"
            "    eor v11.16b, v11.16b, v11.16b\n"

            "    mov x8, %0  // PanelA\n"

            "    cmp %w4, #0       \n"
            "    beq 1f            \n"

            "    mov w19, %w4      \n"

            "    cmp %w3, #0       \n"
            "    beq 2f// loop number is even \n"

            "    // start loopm1_kd8_nd4\n"
            "    subs w19, w19, #1        \n"
            "    ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [%1], #32 // load four lines of B\n"
            "    ld1 {v2.8b}, [%0], #8  // load two lines of PanelA\n"

            "    smull v0.8h, v4.8b, v2.8b  \n"
            "    saddlp v8.4s, v0.8h        \n"
            "    smull v0.8h, v5.8b, v2.8b  \n"
            "    saddlp v9.4s, v0.8h        \n"
            "    smull v0.8h, v6.8b, v2.8b  \n"
            "    saddlp v10.4s, v0.8h       \n"
            "    smull v0.8h, v7.8b, v2.8b  \n"
            "    saddlp v11.4s, v0.8h       \n"

            "    cmp w19, #0            \n"
            "    beq 3f                 \n"

            "    2:  \n"
            "        ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [%1], #32    \n"
            "        ld1 {v12.8b, v13.8b, v14.8b, v15.8b}, [%1], #32\n"

            "        ld1 {v2.8b, v3.8b}, [%0], #16  \n"

            "        smull v0.8h, v2.8b, v4.8b  \n"
            "        smlal v0.8h, v3.8b, v12.8b \n"
            "        sadalp v8.4s, v0.8h        \n"

            "        smull v1.8h, v2.8b, v5.8b  \n"
            "        smlal v1.8h, v3.8b, v13.8b \n"
            "        sadalp v9.4s, v1.8h        \n"

            "        smull v0.8h, v2.8b, v6.8b  \n"
            "        smlal v0.8h, v3.8b, v14.8b \n"
            "        sadalp v10.4s, v0.8h       \n"

            "        smull v1.8h, v2.8b, v7.8b  \n"
            "        smlal v1.8h, v3.8b, v15.8b \n"
            "        sadalp v11.4s, v1.8h       \n"

            "        subs w19, w19, #2          \n"
            "        bne 2b                     \n"

            "    3:                             \n"
            "        addp v8.4s, v8.4s, v9.4s   \n"
            "        addp v10.4s, v10.4s, v11.4s\n"
            "        addp v8.4s, v8.4s, v10.4s  \n"

            "        // start process kd4 kd2 kd1 cases\n"
            "    1:                   \n"
            "        cmp %w5, #0                \n"
            "        beq 4f                     \n"
            "        // start subkernel_m1n4k4  \n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  // load B4x4\n"
            "        sxtl v4.8h, v4.8b          \n"
            "        sxtl v5.8h, v5.8b          \n"
            "        mov v6.d[0], v4.d[1]       \n"
            "        mov v7.d[0], v5.d[1]       \n"

            "        ld1 {v2.8b}, [%0]  // load A1x4\n"
            "        add %0, %0, #4                 \n"
            "        sxtl v2.8h, v2.8b              \n"

            "        smull v12.4s, v2.4h, v4.4h \n"
            "        smull v13.4s, v2.4h, v6.4h \n"
            "        smull v14.4s, v2.4h, v5.4h \n"
            "        smull v15.4s, v2.4h, v7.4h \n"

            "        addp v12.4s, v12.4s, v13.4s\n"
            "        addp v14.4s, v14.4s, v15.4s\n"
            "        addp v12.4s, v12.4s, v14.4s\n"
            "        add v8.4s, v8.4s, v12.4s   \n"

            "    4:                        \n"
            "        cmp %w6, #0            \n"
            "        beq 5f                 \n"
            "        // start subkernel_m1n4k2\n"
            "        ld1 {v4.8b}, [%0]       // load A1x2   \n"
            "        add %0, %0, #2                         \n"
            "        ld1 {v0.8b}, [%1], #8   // load B2x4   \n"

            "        mov v4.h[1], v4.h[0]   \n"
            "        mov v4.s[1], v4.s[0]   \n"

            "        smull v0.8h, v0.8b, v4.8b  \n"
            "        sadalp v8.4s, v0.8h        \n"

            "    5:                    \n"
            "        cmp %w7, #0       \n"
            "        beq 6f            \n"
            "        // start subkernel_m1n4k1  \n"
            "        ld1 {v4.8b}, [%1]   // load B1x4\n"
            "        add %1, %1, #4     \n"
            "        ld1 {v2.8b}, [%0]   // load A1x1\n"
            "        add %0, %0, #1     \n"
            "        sxtl v4.8h, v4.8b  \n"
            "        sxtl v2.8h, v2.8b  \n"
            "        smlal v8.4s, v4.4h, v2.h[0]\n"

            "    6:                             \n"
            "        cmp %9, #0                 \n"
            "        beq 7f                     \n"
            "        ldr w24, [%9]              \n"
            "        // int32 => fp32           \n"
            "        scvtf v8.4s, v8.4s         \n"
            "        // fp32 *= scale_tm        \n"
            "        mov v12.s[0], w24          \n"
            "        fmul v8.4s, v8.4s, v12.s[0]\n"

            "        cmp %10, #0                \n"
            "        beq 8f                     \n"

            "        // fp32 += bias_tm         \n"
            "        ldr w24, [%10]             \n"
            "        dup v15.4s, w24            \n"
            "        fadd v8.4s, v8.4s, v15.4s  \n"

            "        8:                     \n"
            "            // fp32 -> int32   \n"
            "            fcvtas v8.4s, v8.4s\n"
            "            // int32 -> int16  \n"
            "            sqxtn v8.4h, v8.4s \n"
            "            // int16 -> int8   \n"
            "            sqxtn v8.8b, v8.8h \n"
            "            // save            \n"
            "            st1 {v8.s}[0], [%2]\n"
            "            add %2, %2, #4     \n"
            "            b 10f\n"
            "    7:                         \n"
            "        st1 {v8.4s}, [%2], #16 \n"

            "    10:     \n"
            "        subs %w8, %w8, #1  \n"
            "        mov %0, x8         \n"
            "        bne 9b             \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc),      // %2
            "=r"(k8_even), // %3
            "=r"(k8),      // %4
            "=r"(k4),      // %5
            "=r"(k2),      // %6
            "=r"(k1),      // %7
            "=r"(n4),      // %8
            "=r"(scales),  // %9
            "=r"(bias)     // %10
            : "0"(pa),
            "1"(pb),
            "2"(pc),
            "3"(k8_even),
            "4"(k8),
            "5"(k4),
            "6"(k2),
            "7"(k1),
            "8"(n4),
            "9"(scales),
            "10"(bias)
            : "cc", "memory", "x8", "w19", "w24", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }

    if (n2 > 0)
    {
        asm volatile(
            "9:                  \n"
            "    eor v8.16b, v8.16b, v8.16b   \n"
            "    eor v9.16b, v9.16b, v9.16b   \n"
            "    eor v10.16b, v10.16b, v10.16b\n"
            "    eor v11.16b, v11.16b, v11.16b\n"

            "    mov x8, %0  // PanelA\n"

            "    cmp %w4, #0                  \n"
            "    beq 1f     // k <= 7\n"

            "    mov w19, %w4\n"
            "    cmp %w3, #0 \n"
            "    beq 2f     // loop number is even \n"

            "    // start loopmd1_kd8_nd2   \n"
            "    subs w19, w19, #1          \n"
            "    ld1 {v4.8b, v5.8b}, [%1], #16  // load two lines of B\n"
            "    ld1 {v2.8b}, [%0], #8  // load two lines of PanelA\n"
            "    smull v0.8h, v4.8b, v2.8b  \n"
            "    saddlp v8.4s, v0.8h        \n"
            "    smull v0.8h, v5.8b, v2.8b  \n"
            "    saddlp v9.4s, v0.8h        \n"

            "    cmp w19, #0        \n"
            "    beq 3f             \n"

            "    2:  \n"
            "        ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [%1], #32\n"

            "        ld1 {v2.8b, v3.8b}, [%0], #16  \n"

            "        smull v0.8h, v2.8b, v4.8b  \n"
            "        smlal v0.8h, v3.8b, v6.8b  \n"
            "        sadalp v8.4s, v0.8h        \n"

            "        smull v1.8h, v2.8b, v5.8b  \n"
            "        smlal v1.8h, v3.8b, v7.8b  \n"
            "        sadalp v9.4s, v1.8h        \n"

            "        subs w19, w19, #2      \n"
            "        bne 2b                 \n"

            "    3:                             \n"
            "        addp v8.4s, v8.4s, v9.4s   \n"
            "        addp v8.4s, v8.4s, v8.4s   \n"

            "        // start process kd4 kd2 kd1 cases \n"
            "    1:               \n"
            "        cmp %w5, 0                \n"
            "        beq 4f                    \n"
            "        // start subkernel_m1n2k4          \n"
            "        ld1 {v4.8b}, [%1], #8  // load B4x2\n"
            "        sxtl v4.8h, v4.8b      \n"
            "        mov v6.d[0], v4.d[1]   \n"

            "        ld1 {v2.8b}, [%0]      // load A1x4\n"
            "        add %0, %0, #4     \n"
            "        sxtl v2.8h, v2.8b  \n"

            "        smull v9.4s, v2.4h, v4.4h  \n"
            "        smull v10.4s, v2.4h, v6.4h \n"

            "        addp v9.4s, v9.4s, v10.4s  \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        add v8.4s, v8.4s, v9.4s    \n"

            "    4:   \n"
            "        cmp %w6, 0                 \n"
            "        beq 5f                     \n"
            "        // start subkernel_m1n2k2  \n"
            "        ld1 {v4.8b}, [%0]   // load A1x2\n"
            "        add %0, %0, #2 \n"
            "        ld1 {v0.8b}, [%1]   // load B2x2\n"
            "        add %1, %1, #4 \n"

            "        mov v4.h[1], v4.h[0]   \n"

            "        smull v0.8h, v4.8b, v0.8b  \n"
            "        saddlp v0.4s, v0.8h        \n"
            "        add v8.4s, v8.4s, v0.4s    \n"

            "    5:                             \n"
            "        cmp %w7, 0                 \n"
            "        beq 6f                     \n"
            "        // start subkernel_m1n2k1  \n"
            "        ld1 {v4.8b}, [%1]   // load B1x2\n"
            "        add %1, %1, #2             \n"
            "        ld1 {v2.8b}, [%0]   // load A1x1\n"
            "        add %0, %0, #2             \n"
            "        sxtl v4.8h, v4.8b          \n"
            "        sxtl v2.8h, v2.8b          \n"
            "        smlal v8.4s, v4.4h, v2.h[0]\n"

            "    6:                         \n"
            "        cmp %9, #0             \n"
            "        beq 7f                 \n"
            "        // v12: s0 s1          \n"
            "        ldr w24, [%9]          \n"
            "        mov v12.s[0], w24      \n"
            "        mov v12.s[1], v12.s[0] \n"

            "        // int32 => fp32           \n"
            "        scvtf v8.2s, v8.2s         \n"
            "        // fp32 *= scale_tm        \n"
            "        fmul v8.2s, v8.2s, v12.2s  \n"

            "        cmp %10, #0                \n"
            "        beq 8f                     \n"

            "        // fp32 += bias_tm         \n"
            "        ldr w24, [%10]             \n"
            "        mov v12.s[0], w24          \n"
            "        mov v12.s[1], v12.s[0]     \n"
            "        fadd v8.2s, v8.2s, v12.2s  \n"

            "        8:\n"
            "            // fp32 -> int32   \n"
            "            fcvtas v8.2s, v8.2s\n"
            "            // int32 -> int16  \n"
            "            sqxtn v8.4h, v8.4s \n"
            "            // int16 -> int8   \n"
            "            sqxtn v8.8b, v8.8h \n"
            "            // save            \n"
            "            st1 {v8.h}[0], [%2]\n"
            "            add %2, %2, #2     \n"
            "            b 10f\n"
            "    7:          \n"
            "        st1 {v8.2s}, [%2], #8  \n"

            "   10: \n"
            "        mov %0, x8     \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc),      // %2
            "=r"(k8_even), // %3
            "=r"(k8),      // %4
            "=r"(k4),      // %5
            "=r"(k2),      // %6
            "=r"(k1),      // %7
            "=r"(n4),      // %8
            "=r"(scales),  // %9
            "=r"(bias)     // %10
            : "0"(pa),
            "1"(pb),
            "2"(pc),
            "3"(k8_even),
            "4"(k8),
            "5"(k4),
            "6"(k2),
            "7"(k1),
            "8"(n4),
            "9"(scales),
            "10"(bias)
            : "cc", "memory", "x8", "w19", "w24", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }

    if (n1 > 0)
    {
        asm volatile(
            "9:                    \n"
            "    eor v8.16b, v8.16b, v8.16b   \n"
            "    eor v9.16b, v9.16b, v9.16b   \n"
            "    eor v10.16b, v10.16b, v10.16b\n"
            "    eor v11.16b, v11.16b, v11.16b\n"

            "    cmp %w4, #0                    \n"
            "    beq 1f // k <= 7   \n"

            "    mov w19, %w4\n"

            "    cmp %w3, #0 \n"
            "    beq 2f // loop number is even \n"

            "    // start loopkd8_nd1                   \n"
            "    subs w19, w19, #1                      \n"
            "    ld1 {v4.8b}, [%1], #8  // load B line  \n"
            "    ld1 {v2.8b}, [%0], #8  // load A line  \n"
            "    smull v0.8h, v4.8b, v2.8b              \n"
            "    saddlp v8.4s, v0.8h                    \n"

            "    cmp w19, #0           \n"
            "    beq 3f                \n"

            "    2:                  \n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  \n"
            "        ld1 {v24.8b, v25.8b}, [%0], #16\n"

            "        smull v0.8h, v24.8b, v4.8b \n"
            "        smlal v0.8h, v25.8b, v5.8b \n"
            "        sadalp v8.4s, v0.8h        \n"

            "        subs w19, w19, #2          \n"
            "        bne 2b                     \n"

            "    3:                             \n"
            "        addp v8.4s, v8.4s, v8.4s   \n"
            "        addp v8.4s, v8.4s, v8.4s   \n"

            "        // start process kd4 kd2 kd1 cases\n"
            "    1:                \n"
            "        cmp %w5, 0                 \n"
            "        beq 4f                     \n"
            "        // start subkernel_m1n1k4  \n"
            "        ld1 {v4.8b}, [%1]  // load B4x1\n"
            "        add %1, %1, #4             \n"
            "        sxtl v4.8h, v4.8b   // extend B4x1 to v4\n"

            "        ld1 {v2.8b}, [%0]  // load A1x4\n"
            "        add %0, %0, #4             \n"
            "        sxtl v2.8h, v2.8b          \n"

            "        smull v9.4s, v2.4h, v4.4h  \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        add v8.4s, v8.4s, v9.4s    \n"

            "    4:   \n"
            "        cmp %w6, 0    \n"
            "        beq 5f                     \n"
            "        // start subkernel_m1n1k2  \n"
            "        ld1 {v4.8b}, [%0]   // load A1x2\n"
            "        add %0, %0, #2             \n"
            "        ld1 {v0.8b}, [%1]   // load B2x1\n"
            "        add %1, %1, #2             \n"

            "        smull v0.8h, v0.8b, v4.8b  \n"
            "        saddlp v0.4s, v0.8h        \n"

            "        add v8.4s, v8.4s, v0.4s    \n"

            "    5:                \n"
            "        cmp %w7, 0                 \n"
            "        beq 6f                     \n"
            "        // start subkernel_m1n1k1  \n"

            "        ld1 {v0.8b}, [%1]    // load B1x1  \n"
            "        add %1, %1, #1                     \n"

            "        ld1 {v1.8b}, [%0]   // load A1x1   \n"
            "        add %0, %0, #1                     \n"

            "        sxtl v1.8h, v1.8b  \n"
            "        sxtl v0.8h, v0.8b  \n"

            "        smull v0.4s, v1.4h, v0.h[0]    \n"

            "        add v8.4s, v8.4s, v0.4s    \n"

            "    6:            \n"
            "        cmp %9, #0             \n"
            "        beq 7f             \n"
            "        // int32 => fp32   \n"
            "        scvtf v8.2s, v8.2s \n"
            "        // fp32 *= scale_tm\n"
            "        ldr w24, [%9]      \n"
            "        mov v12.s[0], w24  \n"
            "        fmul v8.2s, v8.2s, v12.2s \n"

            "        cmp %10, #0                \n"
            "        beq 8f                     \n"

            "        // fp32 += bias_tm         \n"
            "        ldr w24, [%10]             \n"
            "        mov v12.s[0], w24          \n"
            "        fadd v8.2s, v8.2s, v12.2s  \n"

            "        8:                     \n"
            "            // fp32 -> int32   \n"
            "            fcvtas v8.2s, v8.2s\n"
            "            // int32 -> int16  \n"
            "            sqxtn v8.4h, v8.4s \n"
            "            // int16 -> int8   \n"
            "            sqxtn v8.8b, v8.8h \n"
            "            // save            \n"
            "            st1 {v8.b}[0], [%2]\n"
            "        b 10f                  \n"

            "    7:          \n"
            "        st1 {v8.s}[0], [%2]    \n"
            "    10:                        \n"
            "        mov x0, #0             \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc),      // %2
            "=r"(k8_even), // %3
            "=r"(k8),      // %4
            "=r"(k4),      // %5
            "=r"(k2),      // %6
            "=r"(k1),      // %7
            "=r"(n4),      // %8
            "=r"(scales),  // %9
            "=r"(bias)     // %10
            : "0"(pa),
            "1"(pb),
            "2"(pc),
            "3"(k8_even),
            "4"(k8),
            "5"(k4),
            "6"(k2),
            "7"(k1),
            "8"(n4),
            "9"(scales),
            "10"(bias)
            : "cc", "memory", "x0", "x8", "w19", "w24", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }
}

static void int8kernel_m2(void* dst, int8_t* sa, int8_t* sb, int, int k, int n, int ldc, float* scales, float* bias)
{
    void *pc0, *pc1;
    if (scales == 0)
    {
        pc0 = (int32_t*)dst;
        pc1 = ((int32_t*)pc0) + ldc;
    }
    else
    {
        pc0 = dst;
        pc1 = ((int8_t*)pc0) + ldc;
    }
    int8_t* pa = sa;
    int8_t* pb = sb;

    DECOMPOSE_K
    DECOMPOSE_N

    if (n4 > 0)
    {
        asm volatile(
            "9:                        \n"
            "    eor v8.16b, v8.16b, v8.16b     \n"
            "    eor v9.16b, v9.16b, v9.16b     \n"
            "    eor v10.16b, v10.16b, v10.16b  \n"
            "    eor v11.16b, v11.16b, v11.16b  \n"
            "    eor v12.16b, v12.16b, v12.16b  \n"
            "    eor v13.16b, v13.16b, v13.16b  \n"
            "    eor v14.16b, v14.16b, v14.16b  \n"
            "    eor v15.16b, v15.16b, v15.16b  \n"

            "    eor v16.16b, v16.16b, v16.16b  \n"
            "    eor v17.16b, v17.16b, v17.16b  \n"
            "    eor v18.16b, v18.16b, v18.16b  \n"
            "    eor v19.16b, v19.16b, v19.16b  \n"
            "    eor v20.16b, v20.16b, v20.16b  \n"
            "    eor v21.16b, v21.16b, v21.16b  \n"
            "    eor v22.16b, v22.16b, v22.16b  \n"
            "    eor v23.16b, v23.16b, v23.16b  \n"

            "    mov x8, %0  // PanelA  \n"

            "    cmp %w5, #0            \n"
            "    beq 1f                 \n"

            "    mov w17, %w5           \n"

            "    cmp %w4, #0            \n"
            "    beq 2f // loop number is even \n"

            "    // start loopm2_kd8_nd4\n"
            "    subs w17, w17, #1      \n"
            "    ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [%1], #32 // load four lines of B\n"
            "    ld1 {v2.8b, v3.8b}, [%0], #16  // load two lines of PanelA         \n"
            "    smull v0.8h, v4.8b, v2.8b  \n"
            "    smull v1.8h, v4.8b, v3.8b  \n"
            "    saddlp v8.4s, v0.8h        \n"
            "    saddlp v12.4s, v1.8h       \n"
            "    smull v0.8h, v5.8b, v2.8b  \n"
            "    smull v1.8h, v5.8b, v3.8b  \n"
            "    saddlp v9.4s, v0.8h        \n"
            "    saddlp v13.4s, v1.8h       \n"
            "    smull v0.8h, v6.8b, v2.8b  \n"
            "    smull v1.8h, v6.8b, v3.8b  \n"
            "    saddlp v10.4s, v0.8h       \n"
            "    saddlp v14.4s, v1.8h       \n"
            "    smull v0.8h, v7.8b, v2.8b  \n"
            "    smull v1.8h, v7.8b, v3.8b  \n"
            "    saddlp v11.4s, v0.8h       \n"
            "    saddlp v15.4s, v1.8h       \n"

            "    cmp w17, #0                \n"
            "    beq 3f                     \n"

            "    2:               \n"
            "        add x12, %1, #32               \n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  \n"
            "        ld1 {v2.8b, v3.8b}, [%0], #16  \n"

            "        smull v0.8h, v4.8b, v2.8b      \n"
            "        smull v1.8h, v5.8b, v2.8b      \n"

            "        ld1 {v6.8b, v7.8b}, [x12], #16 \n"
            "        ld1 {v24.8b, v25.8b}, [%0], #16\n"

            "        smlal v0.8h, v6.8b, v24.8b \n"
            "        smlal v1.8h, v7.8b, v24.8b \n"

            "        sadalp v8.4s, v0.8h\n"
            "        sadalp v9.4s, v1.8h\n"

            "        smull v0.8h, v4.8b, v3.8b  \n"
            "        smull v1.8h, v5.8b, v3.8b  \n"
            "        smlal v0.8h, v6.8b, v25.8b \n"
            "        smlal v1.8h, v7.8b, v25.8b \n"

            "        sadalp v12.4s, v0.8h\n"
            "        sadalp v13.4s, v1.8h\n"

            "        // start v10v11, v14v15, v18v19, v22v23, error here!\n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  \n"
            "        smull v0.8h, v4.8b, v2.8b      \n"
            "        smull v1.8h, v5.8b, v2.8b      \n"
            "        ld1 {v6.8b, v7.8b}, [x12], #16 \n"
            "        smlal v0.8h, v6.8b, v24.8b     \n"
            "        smlal v1.8h, v7.8b, v24.8b     \n"
            "        sadalp v10.4s, v0.8h           \n"
            "        sadalp v11.4s, v1.8h           \n"

            "        smull v0.8h, v4.8b, v3.8b  \n"
            "        smull v1.8h, v5.8b, v3.8b  \n"
            "        smlal v0.8h, v6.8b, v25.8b \n"
            "        smlal v1.8h, v7.8b, v25.8b \n"
            "        sadalp v14.4s, v0.8h       \n"
            "        sadalp v15.4s, v1.8h       \n"

            "        add %1, %1, #32        \n"
            "        subs w17, w17, #2      \n"
            "        bne 2b                 \n"

            "    3:               \n"
            "        addp v8.4s, v8.4s, v9.4s   \n"
            "        addp v10.4s, v10.4s, v11.4s\n"
            "        addp v12.4s, v12.4s, v13.4s\n"
            "        addp v14.4s, v14.4s, v15.4s\n"

            "        addp v8.4s, v8.4s, v10.4s  \n"
            "        addp v9.4s, v12.4s, v14.4s \n"

            "        // start process kd4 kd2 kd1 cases \n"
            "    1:                \n"
            "        cmp %w6, #0                \n"
            "        beq 4f                     \n"
            "        // start subkernel_m2n4k4  \n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  // load B4x4\n"
            "        sxtl v4.8h, v4.8b          \n"
            "        sxtl v5.8h, v5.8b          \n"
            "        mov v6.d[0], v4.d[1]       \n"
            "        mov v7.d[0], v5.d[1]       \n"

            "        ld1 {v2.8b}, [%0], #8  // load A2x4\n"
            "        sxtl v2.8h, v2.8b          \n"
            "        mov v3.d[0], v2.d[1]       \n"

            "        smull v12.4s, v2.4h, v4.4h \n"
            "        smull v13.4s, v2.4h, v6.4h \n"
            "        smull v14.4s, v2.4h, v5.4h \n"
            "        smull v15.4s, v2.4h, v7.4h \n"

            "        addp v12.4s, v12.4s, v13.4s\n"
            "        addp v14.4s, v14.4s, v15.4s\n"
            "        addp v12.4s, v12.4s, v14.4s\n"
            "        add v8.4s, v8.4s, v12.4s   \n"

            "        smull v16.4s, v3.4h, v4.4h \n"
            "        smull v17.4s, v3.4h, v6.4h \n"
            "        smull v18.4s, v3.4h, v5.4h \n"
            "        smull v19.4s, v3.4h, v7.4h \n"

            "        addp v16.4s, v16.4s, v17.4s\n"
            "        addp v18.4s, v18.4s, v19.4s\n"
            "        addp v16.4s, v16.4s, v18.4s\n"
            "        add v9.4s, v9.4s, v16.4s   \n"

            "    4:                \n"
            "        cmp %w7, #0                \n"
            "        beq 5f                     \n"
            "        // start subkernel_m2n4k2  \n"
            "        ld1 {v4.8b}, [%0]       // load A2x2   \n"
            "        add %0, %0, #4             \n"
            "        ld1 {v0.8b}, [%1], #8   // load B2x4   \n"
            "                            // 00 11 22 33 \n"
            "        rev32 v1.4h, v0.4h  // 11 00 33 22 \n"
            "        rev64 v2.2s, v0.2s  // 22 33 00 11 \n"
            "        rev64 v3.4h, v0.4h  // 33 22 11 00 \n"

            "        smull v12.8h, v4.8b, v0.8b \n"
            "        smull v13.8h, v4.8b, v1.8b \n"
            "        smull v14.8h, v4.8b, v2.8b \n"
            "        smull v15.8h, v4.8b, v3.8b \n"

            "        saddlp v12.4s, v12.8h  \n"
            "        saddlp v13.4s, v13.8h  \n"
            "        saddlp v14.4s, v14.8h  \n"
            "        saddlp v15.4s, v15.8h  \n"

            "        mov v16.s[0], v12.s[0] \n"
            "        mov v16.s[1], v13.s[0] \n"
            "        mov v16.s[2], v14.s[0] \n"
            "        mov v16.s[3], v15.s[0] \n"

            "        mov v17.s[0], v13.s[1] \n"
            "        mov v17.s[1], v12.s[1] \n"
            "        mov v17.s[2], v15.s[1] \n"
            "        mov v17.s[3], v14.s[1] \n"

            "        add v8.4s, v8.4s, v16.4s   \n"
            "        add v9.4s, v9.4s, v17.4s   \n"

            "    5:            \n"
            "        cmp %w8, #0            \n"
            "        beq 6f                 \n"
            "        // start subkernel_m2n4k1  \n"
            "        ld1 {v4.8b}, [%1]   // load B1x4\n"
            "        add %1, %1, #4         \n"
            "        ld1 {v2.8b}, [%0]   // load A2x1\n"
            "        add %0, %0, #2         \n"
            "        sxtl v4.8h, v4.8b      \n"
            "        sxtl v2.8h, v2.8b      \n"
            "        smlal v8.4s, v4.4h, v2.h[0]\n"
            "        smlal v9.4s, v4.4h, v2.h[1]\n"

            "    6:                        \n"
            "        cmp %10, #0           \n"
            "        beq 7f                 \n"

            "        ld1 {v12.2s}, [%10]        \n"
            "        // int32 => fp32           \n"
            "        scvtf v8.4s, v8.4s         \n"
            "        scvtf v9.4s, v9.4s         \n"
            "        // fp32 *= scale_tm        \n"
            "        fmul v8.4s, v8.4s, v12.s[0]\n"
            "        fmul v9.4s, v9.4s, v12.s[1]\n"

            "        cmp %11, #0                \n"
            "        beq 8f                     \n"

            "        // fp32 += scales_tm       \n"
            "        ld1 {v14.2s}, [%11]        \n"
            "        dup v15.4s, v14.s[0]       \n"
            "        fadd v8.4s, v8.4s, v15.4s  \n"
            "        dup v15.4s, v14.s[1]       \n"
            "        fadd v9.4s, v9.4s, v15.4s  \n"

            "        8:                     \n"
            "            // fp32 -> int32   \n"
            "            fcvtas v8.4s, v8.4s\n"
            "            fcvtas v9.4s, v9.4s\n"
            "            // int32 -> int16  \n"
            "            sqxtn v6.4h, v8.4s \n"
            "            sqxtn2 v6.8h, v9.4s\n"
            "            // int16 -> int8   \n"
            "            sqxtn v8.8b, v6.8h \n"
            "            // save                \n"
            "            st1 {v8.s}[0], [%2]    \n"
            "            add %2, %2, #4         \n"
            "            st1 {v8.s}[1], [%3]    \n"
            "            add %3, %3, #4         \n"
            "            b 10f  \n"
            "    7:              \n"
            "        st1 {v8.4s}, [%2], #16     \n"
            "        st1 {v9.4s}, [%3], #16     \n"

            "   10:      \n"
            "        subs %w9, %w9, #1    \n"
            "        mov %0, x8           \n"
            "        bne 9b               \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc0),     // %2
            "=r"(pc1),     // %3
            "=r"(k8_even), // %4
            "=r"(k8),      // %5
            "=r"(k4),      // %6
            "=r"(k2),      // %7
            "=r"(k1),      // %8
            "=r"(n4),      // %9
            "=r"(scales),  // %10
            "=r"(bias)     // %11
            : "0"(pa),
            "1"(pb),
            "2"(pc0),
            "3"(pc1),
            "4"(k8_even),
            "5"(k8),
            "6"(k4),
            "7"(k2),
            "8"(k1),
            "9"(n4),
            "10"(scales),
            "11"(bias)
            : "cc", "memory", "x8", "w17", "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }

    if (n2 > 0)
    {
        asm volatile(
            "eor v8.16b, v8.16b, v8.16b     \n"
            "eor v9.16b, v9.16b, v9.16b     \n"
            "eor v10.16b, v10.16b, v10.16b  \n"
            "eor v11.16b, v11.16b, v11.16b  \n"
            "eor v12.16b, v12.16b, v12.16b  \n"
            "eor v13.16b, v13.16b, v13.16b  \n"
            "eor v14.16b, v14.16b, v14.16b  \n"
            "eor v15.16b, v15.16b, v15.16b  \n"

            "eor v16.16b, v16.16b, v16.16b  \n"
            "eor v17.16b, v17.16b, v17.16b  \n"
            "eor v18.16b, v18.16b, v18.16b  \n"
            "eor v19.16b, v19.16b, v19.16b  \n"
            "eor v20.16b, v20.16b, v20.16b  \n"
            "eor v21.16b, v21.16b, v21.16b  \n"
            "eor v22.16b, v22.16b, v22.16b  \n"
            "eor v23.16b, v23.16b, v23.16b  \n"

            "9:                \n"
            "    mov x8, %0  // PanelA      \n"

            "    cmp %w5, #0                \n"
            "    beq 1f                     \n"

            "    mov w17, %w5    \n"
            "    cmp %w4, #0     \n"
            "    beq 2f         // loop number is even \n"

            "    // start loopmd2_kd8_nd2   \n"
            "    subs w17, w17, #1          \n"
            "    ld1 {v4.8b, v5.8b}, [%1], #16  // load two lines of B\n"
            "    ld1 {v2.8b, v3.8b}, [%0], #16  // load two lines of PanelA\n"
            "    smull v0.8h, v4.8b, v2.8b  \n"
            "    smull v1.8h, v4.8b, v3.8b  \n"
            "    saddlp v8.4s, v0.8h        \n"
            "    saddlp v12.4s, v1.8h       \n"
            "    smull v0.8h, v5.8b, v2.8b  \n"
            "    smull v1.8h, v5.8b, v3.8b  \n"
            "    saddlp v9.4s, v0.8h        \n"
            "    saddlp v13.4s, v1.8h       \n"

            "    cmp w17, #0        \n"
            "    beq 3f             \n"

            "    2:  \n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  \n"
            "        ld1 {v2.8b, v3.8b}, [%0], #16  \n"

            "        smull v0.8h, v4.8b, v2.8b      \n"
            "        ld1 {v6.8b, v7.8b}, [%1], #16  \n"
            "        smull v1.8h, v5.8b, v2.8b      \n"
            "        ld1 {v24.8b, v25.8b}, [%0], #16\n"

            "        smlal v0.8h, v6.8b, v24.8b \n"
            "        smlal v1.8h, v7.8b, v24.8b \n"

            "        sadalp v8.4s, v0.8h\n"
            "        sadalp v9.4s, v1.8h\n"

            "        smull v0.8h, v4.8b, v3.8b  \n"
            "        smull v1.8h, v5.8b, v3.8b  \n"
            "        smlal v0.8h, v6.8b, v25.8b \n"
            "        smlal v1.8h, v7.8b, v25.8b \n"

            "        sadalp v12.4s, v0.8h       \n"
            "        sadalp v13.4s, v1.8h       \n"

            "        subs w17, w17, #2          \n"
            "        bne 2b                     \n"

            "    3:            \n"
            "        addp v8.4s, v8.4s, v9.4s   \n"
            "        addp v12.4s, v12.4s, v13.4s\n"

            "        addp v8.4s, v8.4s, v8.4s   \n"
            "        addp v12.4s, v12.4s, v12.4s\n"

            "        // start process kd4 kd2 kd1 cases\n"
            "    1:                \n"
            "        cmp %w6, #0                \n"
            "        beq 4f                     \n"
            "        // start subkernel_m2n2k4  \n"
            "        ld1 {v4.8b}, [%1], #8  // load B4x2\n"
            "        sxtl v4.8h, v4.8b          \n"
            "        mov v6.d[0], v4.d[1]       \n"

            "        ld1 {v2.8b}, [%0], #8  // load first A2x4\n"
            "        sxtl v2.8h, v2.8b          \n"
            "        mov v3.d[0], v2.d[1]       \n"

            "        smull v9.4s, v2.4h, v4.4h  \n"
            "        smull v10.4s, v2.4h, v6.4h \n"

            "        addp v9.4s, v9.4s, v10.4s  \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        add v8.4s, v8.4s, v9.4s    \n"

            "        smull v13.4s, v3.4h, v4.4h \n"
            "        smull v14.4s, v3.4h, v6.4h \n"

            "        addp v13.4s, v13.4s, v14.4s\n"
            "        addp v13.4s, v13.4s, v13.4s\n"
            "        add v12.4s, v12.4s, v13.4s \n"

            "    4:                             \n"
            "        cmp %w7, 0                 \n"
            "        beq 5f                     \n"
            "        // start subkernel_m2n2k2  \n"
            "        ld1 {v4.8b}, [%0]   // load A2x2\n"
            "        add %0, %0, #4             \n"
            "        ld1 {v0.8b}, [%1]   // load B2x2\n"
            "        add %1, %1, #4             \n"
            "                            // 00 11\n"
            "        rev32 v1.4h, v0.4h  // 11 00\n"

            "        smull v21.8h, v4.8b, v0.8b \n"
            "        smull v22.8h, v4.8b, v1.8b \n"

            "        saddlp v21.4s, v21.8h      \n"
            "        saddlp v22.4s, v22.8h      \n"

            "        mov v9.s[0], v21.s[0]      \n"
            "        mov v9.s[1], v22.s[0]      \n"
            "        add v8.4s, v8.4s, v9.4s    \n"

            "        mov v13.s[0], v22.s[1]     \n"
            "        mov v13.s[1], v21.s[1]     \n"
            "        add v12.4s, v12.4s, v13.4s \n"

            "    5:                \n"
            "        cmp %w8, #0                \n"
            "        beq 6f                     \n"
            "        // start subkernel_m2n2k1  \n"
            "        ld1 {v4.8b}, [%1]   // load B1x2\n"
            "        add %1, %1, #2             \n"
            "        ld1 {v2.8b}, [%0]   // load A4x1\n"
            "        add %0, %0, #2             \n"
            "        sxtl v4.8h, v4.8b          \n"
            "        sxtl v2.8h, v2.8b          \n"
            "        smlal v8.4s, v4.4h, v2.h[0]\n"
            "        smlal v12.4s, v4.4h, v2.h[1]   \n"

            "    6:               \n"
            "        cmp %9, #0                \n"
            "        beq 7f                     \n"

            "        mov v8.d[1], v12.d[0]     \n"

            "        // v12: 0 1                \n"
            "        ld1 {v12.2s}, [%9]         \n"
            "        zip1 v12.4s, v12.4s, v12.4s\n"
            "        // v12: 0 0 1 1            \n"

            "        // int32 => fp32           \n"
            "        scvtf v8.4s, v8.4s         \n"
            "        // fp32 *= scale_tm        \n"
            "        fmul v8.4s, v8.4s, v12.4s  \n"

            "        cmp %10, #0                \n"
            "        beq 8f                     \n"

            "        // fp32 += bias_tm         \n"
            "        ld1 {v12.2s}, [%10]        \n"
            "        zip1 v12.4s, v12.4s, v12.4s\n"
            "        fadd v8.4s, v8.4s, v12.4s  \n"

            "        8:    \n"
            "            // fp32 -> int32       \n"
            "            fcvtas v8.4s, v8.4s    \n"
            "            // int32 -> int16      \n"
            "            sqxtn v8.4h, v8.4s     \n"
            "            // int16 -> int8       \n"
            "            sqxtn v8.8b, v8.8h     \n"
            "            // save                \n"
            "            st1 {v8.h}[0], [%2]    \n"
            "            add %2, %2, #2         \n"
            "            st1 {v8.h}[1], [%3]    \n"
            "            add %3, %3, #2         \n"
            "            b 10f   \n"

            "    7:"
            "        st1 {v8.2s}, [%2], #8      \n"
            "        st1 {v12.2s}, [%3], #8     \n"
            "    10:            \n"
            "        mov %0, x8                 \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc0),     // %2
            "=r"(pc1),     // %3
            "=r"(k8_even), // %4
            "=r"(k8),      // %5
            "=r"(k4),      // %6
            "=r"(k2),      // %7
            "=r"(k1),      // %8
            "=r"(scales),  // %9
            "=r"(bias)     // %10
            : "0"(pa),
            "1"(pb),
            "2"(pc0),
            "3"(pc1),
            "4"(k8_even),
            "5"(k8),
            "6"(k4),
            "7"(k2),
            "8"(k1),
            "9"(scales),
            "10"(bias)
            : "cc", "memory", "x8", "x12", "w17", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }

    if (n1 > 0)
    {
        asm volatile(
            "eor v8.16b, v8.16b, v8.16b     \n"
            "eor v9.16b, v9.16b, v9.16b     \n"
            "eor v10.16b, v10.16b, v10.16b  \n"
            "eor v11.16b, v11.16b, v11.16b  \n"
            "eor v12.16b, v12.16b, v12.16b  \n"
            "eor v13.16b, v13.16b, v13.16b  \n"
            "eor v14.16b, v14.16b, v14.16b  \n"
            "eor v15.16b, v15.16b, v15.16b  \n"

            "eor v16.16b, v16.16b, v16.16b  \n"
            "eor v17.16b, v17.16b, v17.16b  \n"
            "eor v18.16b, v18.16b, v18.16b  \n"
            "eor v19.16b, v19.16b, v19.16b  \n"
            "eor v20.16b, v20.16b, v20.16b  \n"
            "eor v21.16b, v21.16b, v21.16b  \n"
            "eor v22.16b, v22.16b, v22.16b  \n"
            "eor v23.16b, v23.16b, v23.16b  \n"

            "9:                \n"
            "    cmp %w5, #0                \n"
            "    beq 1f             // k <=7\n"

            "    mov w17, %w5\n"

            "    cmp %w4, #0 \n"
            "    beq 2f     // loop number is even \n"

            "    // start loopkd8_nd1   \n"
            "    subs w17, w17, #1      \n"
            "    ld1 {v4.8b}, [%1], #8  // load four lines of B\n"
            "    ld1 {v2.8b, v3.8b}, [%0], #16  // load two lines of PanelA\n"
            "    smull v0.8h, v4.8b, v2.8b  \n"
            "    smull v1.8h, v4.8b, v3.8b  \n"
            "    saddlp v8.4s, v0.8h        \n"
            "    saddlp v12.4s, v1.8h       \n"

            "    cmp w17, #0           \n"
            "    beq 3f                \n"

            "    2:  \n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16                  \n"
            "        ld1 {v24.8b, v25.8b, v26.8b, v27.8b}, [%0], #32\n"

            "        smull v0.8h, v24.8b, v4.8b \n"
            "        smlal v0.8h, v26.8b, v5.8b \n"
            "        sadalp v8.4s, v0.8h        \n"

            "        smull v1.8h, v25.8b, v4.8b \n"
            "        smlal v1.8h, v27.8b, v5.8b \n"
            "        sadalp v12.4s, v1.8h       \n"

            "        subs w17, w17, #2          \n"
            "        bne 2b                     \n"

            "    3:            \n"
            "        addp v8.4s, v8.4s, v8.4s   \n"
            "        addp v8.4s, v8.4s, v8.4s   \n"
            "        addp v12.4s, v12.4s, v12.4s\n"
            "        addp v12.4s, v12.4s, v12.4s\n"

            "        // start process kd4 kd2 kd1 cases\n"
            "    1:                       \n"
            "        cmp %w6, #0                       \n"
            "        beq 4f                         \n"
            "        // start subkernel_m2n1k2      \n"
            "        ld1 {v4.8b}, [%1]  // load B4x1\n"
            "        add %1, %1, #4                 \n"
            "        sxtl v4.8h, v4.8b   // extend B4x1 to v4\n"

            "        ld1 {v2.8b}, [%0], #8  // load A2x4    \n"
            "        sxtl v2.8h, v2.8b      \n"
            "        mov v5.d[0], v2.d[1]   \n"

            "        smull v9.4s, v2.4h, v4.4h  \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        add v8.4s, v8.4s, v9.4s    \n"

            "        smull v13.4s, v5.4h, v4.4h \n"
            "        addp v13.4s, v13.4s, v13.4s\n"
            "        addp v13.4s, v13.4s, v13.4s\n"
            "        add v12.4s, v12.4s, v13.4s \n"

            "    4:                             \n"
            "        cmp %w7, 0                 \n"
            "        beq 5f                     \n"
            "        // start subkernel_m2n1k2  \n"
            "        ld1 {v4.8b}, [%0]   // load A2x2\n"
            "        add %0, %0, #4             \n"
            "        ld1 {v0.8b}, [%1]   // load B2x1\n"
            "        add %1, %1, #2             \n"

            "        mov v0.h[1], v0.h[0]   \n"

            "        smull v0.8h, v0.8b, v4.8b  \n"
            "        saddlp v0.4s, v0.8h        \n"

            "        mov v9.s[0], v0.s[0]       \n"
            "        add v8.4s, v8.4s, v9.4s    \n"
            "        mov v13.s[0], v0.s[1]      \n"
            "        add v12.4s, v12.4s, v13.4s \n"

            "    5:                             \n"
            "        cmp %w8, 0                 \n"
            "        beq 6f                     \n"
            "        // start subkernel_m2n1k1  \n"

            "        ld1 {v0.8b}, [%1]    // load B1x1\n"
            "        add %1, %1, #1 \n"

            "        ld1 {v1.8b}, [%0]   // load A2x1\n"
            "        add %0, %0, #2 \n"

            "        sxtl v1.8h, v1.8b  \n"
            "        sxtl v0.8h, v0.8b  \n"

            "        smull v0.4s, v1.4h, v0.h[0]\n"
            "        mov v1.s[0], v0.s[1]       \n"

            "        add v8.4s, v8.4s, v0.4s    \n"
            "        add v12.4s, v12.4s, v1.4s  \n"

            "    6:                             \n"
            "        cmp %w9, #0                \n"
            "        beq 7f                     \n"
            "        mov v8.s[1], v12.s[0]      \n"

            "        // v12: s0 s1              \n"
            "        ld1 {v12.2s}, [%9]         \n"
            "        // int32 => fp32           \n"
            "        scvtf v8.2s, v8.2s         \n"
            "        // fp32 *= scale_tm        \n"
            "        fmul v8.2s, v8.2s, v12.2s  \n"

            "        cmp %10, #0                \n"
            "        beq 8f                     \n"

            "        // fp32 += bias_tm         \n"
            "        ld1 {v12.2s}, [%10]        \n"
            "        fadd v8.2s, v8.2s, v12.2s  \n"

            "        8:                         \n"
            "            // fp32 -> int32       \n"
            "            fcvtas v8.2s, v8.2s    \n"
            "            // int32 -> int16      \n"
            "            sqxtn v8.4h, v8.4s     \n"
            "            // int16 -> int8       \n"
            "            sqxtn v8.8b, v8.8h     \n"
            "            // save                \n"
            "            st1 {v8.b}[0], [%2]    \n"
            "            st1 {v8.b}[1], [%3]    \n"
            "            b 10f           \n"

            "    7:                         \n"
            "        st1 {v8.s}[0], [%2]    \n"
            "        st1 {v12.s}[0], [%3]   \n"
            "    10:                \n"
            "        mov x0, #0             \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc0),     // %2
            "=r"(pc1),     // %3
            "=r"(k8_even), // %4
            "=r"(k8),      // %5
            "=r"(k4),      // %6
            "=r"(k2),      // %7
            "=r"(k1),      // %8
            "=r"(scales),  // %9
            "=r"(bias)     // %10
            : "0"(pa),
            "1"(pb),
            "2"(pc0),
            "3"(pc1),
            "4"(k8_even),
            "5"(k8),
            "6"(k4),
            "7"(k2),
            "8"(k1),
            "9"(scales),
            "10"(bias)
            : "cc", "memory", "x0", "x8", "w17", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }
}

static void int8kernel_m4(void* dst, int8_t* sa, int8_t* sb, int, int k, int n, int ldc, float* scales, float* bias)
{
    void *pc0, *pc1, *pc2, *pc3;
    if (scales == 0)
    {
        pc0 = (int32_t*)dst;
        pc1 = ((int32_t*)pc0) + ldc;
        pc2 = ((int32_t*)pc1) + ldc;
        pc3 = ((int32_t*)pc2) + ldc;
    }
    else
    {
        pc0 = dst;
        pc1 = ((int8_t*)pc0) + ldc;
        pc2 = ((int8_t*)pc1) + ldc;
        pc3 = ((int8_t*)pc2) + ldc;
    }
    int8_t* pa = sa;
    int8_t* pb = sb;

    DECOMPOSE_K
    DECOMPOSE_N

    if (n4 > 0)
    {
        asm volatile(
            "8:                             \n"
            "   eor v8.8b, v8.8b, v8.8b    \n"
            "   eor v9.8b, v9.8b, v9.8b    \n"
            "   eor v10.8b, v10.8b, v10.8b \n"
            "   eor v11.8b, v11.8b, v11.8b \n"
            "   eor v12.8b, v12.8b, v12.8b \n"
            "   eor v13.8b, v13.8b, v13.8b \n"
            "   eor v14.8b, v14.8b, v14.8b \n"
            "   eor v15.8b, v15.8b, v15.8b \n"

            "   eor v16.8b, v16.8b, v16.8b \n"
            "   eor v17.8b, v17.8b, v17.8b \n"
            "   eor v18.8b, v18.8b, v18.8b \n"
            "   eor v19.8b, v19.8b, v19.8b \n"
            "   eor v20.8b, v20.8b, v20.8b \n"
            "   eor v21.8b, v21.8b, v21.8b \n"
            "   eor v22.8b, v22.8b, v22.8b \n"
            "   eor v23.8b, v23.8b, v23.8b \n"

            "   mov x8, %0          \n"
            "   cmp %w7, #0         \n"
            "   beq 1f              \n"
            "   mov w20, %w7        \n"

            "   cmp %w6, #0         \n"
            "   beq 2f              \n"

            "   subs w20, w20, #1   \n"
            "   ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [%1], #32 \n"
            "   ld1 {v2.8b, v3.8b}, [%0], #16               \n"
            "   smull v0.8h, v4.8b, v2.8b                   \n"
            "   smull v1.8h, v4.8b, v3.8b                   \n"

            "   saddlp v8.4s, v0.8h             \n"
            "   saddlp v12.4s, v1.8h            \n"
            "   smull v0.8h, v5.8b, v2.8b       \n"
            "   smull v1.8h, v5.8b, v3.8b       \n"
            "   saddlp v9.4s, v0.8h             \n"
            "   saddlp v13.4s, v1.8h            \n"
            "   smull v0.8h, v6.8b, v2.8b       \n"
            "   smull v1.8h, v6.8b, v3.8b       \n"
            "   saddlp v10.4s, v0.8h            \n"
            "   saddlp v14.4s, v1.8h            \n"
            "   smull v0.8h, v7.8b, v2.8b       \n"
            "   smull v1.8h, v7.8b, v3.8b       \n"
            "   saddlp v11.4s, v0.8h            \n"
            "   ld1 {v2.8b, v3.8b}, [%0], #16   \n"
            "   saddlp v15.4s, v1.8h        \n"
            "   smull v0.8h, v4.8b, v2.8b   \n"
            "   smull v1.8h, v4.8b, v3.8b   \n"
            "   saddlp v16.4s, v0.8h        \n"
            "   saddlp v20.4s, v1.8h        \n"
            "   smull v0.8h, v5.8b, v2.8b   \n"
            "   smull v1.8h, v5.8b, v3.8b   \n"
            "   saddlp v17.4s, v0.8h        \n"
            "   saddlp v21.4s, v1.8h        \n"
            "   smull v0.8h, v6.8b, v2.8b   \n"
            "   smull v1.8h, v6.8b, v3.8b   \n"
            "   saddlp v18.4s, v0.8h        \n"
            "   saddlp v22.4s, v1.8h        \n"
            "   smull v0.8h, v7.8b, v2.8b   \n"
            "   smull v1.8h, v7.8b, v3.8b   \n"
            "   saddlp v19.4s, v0.8h        \n"
            "   saddlp v23.4s, v1.8h        \n"

            "   cmp w20, #0                 \n"
            "   beq 3f                      \n"

            "   2:                       \n"
            "       add x15, %x1, #32    \n"
            "       add x14, %x0, #32    \n"
            "       ld1 {v4.8b, v5.8b}, [%1], #16\n"
            "       ld1 {v2.8b, v3.8b}, [%0], #16\n"

            "       smull v0.8h, v4.8b, v2.8b       \n"
            "       ld1 {v6.8b, v7.8b}, [x15], #16  \n"
            "       smull v1.8h, v5.8b, v2.8b       \n"
            "       ld1 {v24.8b, v25.8b}, [x14], #16\n"

            "       smlal v0.8h, v6.8b, v24.8b\n"
            "       smlal v1.8h, v7.8b, v24.8b\n"

            "       sadalp v8.4s, v0.8h\n"
            "       sadalp v9.4s, v1.8h\n"

            "       smull v0.8h, v4.8b, v3.8b   \n"
            "       smull v1.8h, v5.8b, v3.8b   \n"
            "       smlal v0.8h, v6.8b, v25.8b  \n"
            "       smlal v1.8h, v7.8b, v25.8b  \n"

            "       sadalp v12.4s, v0.8h\n"
            "       sadalp v13.4s, v1.8h\n"

            "       // finish v8v9 v12v13, start proc v16v17,v20v21\n"
            "       ld1 {v28.8b, v29.8b}, [%0], #16 \n"
            "       smull v0.8h, v4.8b, v28.8b      \n"
            "       smull v1.8h, v5.8b, v28.8b      \n"
            "       ld1 {v26.8b, v27.8b}, [x14], #16\n"
            "       smlal v0.8h, v6.8b, v26.8b      \n"
            "       smlal v1.8h, v7.8b, v26.8b      \n"
            "       sadalp v16.4s, v0.8h            \n"
            "       sadalp v17.4s, v1.8h            \n"

            "       smull v0.8h, v4.8b, v29.8b  \n"
            "       smull v1.8h, v5.8b, v29.8b  \n"
            "       smlal v0.8h, v6.8b, v27.8b  \n"
            "       smlal v1.8h, v7.8b, v27.8b  \n"
            "       sadalp v20.4s, v0.8h        \n"
            "       sadalp v21.4s, v1.8h        \n"

            "       // start v10v11, v14v15, v18v19, v22v23\n"
            "       ld1 {v4.8b, v5.8b}, [%1], #16   \n"
            "       smull v0.8h, v4.8b, v2.8b       \n"
            "       smull v1.8h, v5.8b, v2.8b       \n"
            "       ld1 {v6.8b, v7.8b}, [x15], #16  \n"
            "       smlal v0.8h, v6.8b, v24.8b      \n"
            "       smlal v1.8h, v7.8b, v24.8b      \n"
            "       sadalp v10.4s, v0.8h            \n"
            "       sadalp v11.4s, v1.8h            \n"

            "       smull v0.8h, v4.8b, v3.8b   \n"
            "       smull v1.8h, v5.8b, v3.8b   \n"
            "       smlal v0.8h, v6.8b, v25.8b  \n"
            "       smlal v1.8h, v7.8b, v25.8b  \n"
            "       sadalp v14.4s, v0.8h        \n"
            "       sadalp v15.4s, v1.8h        \n"

            "       smull v0.8h, v4.8b, v28.8b  \n"
            "       smull v1.8h, v5.8b, v28.8b  \n"
            "       smlal v0.8h, v6.8b, v26.8b  \n"
            "       smlal v1.8h, v7.8b, v26.8b  \n"
            "       sadalp v18.4s, v0.8h        \n"
            "       sadalp v19.4s, v1.8h        \n"

            "       smull v0.8h, v4.8b, v29.8b  \n"
            "       smull v1.8h, v5.8b, v29.8b  \n"
            "       smlal v0.8h, v6.8b, v27.8b  \n"
            "       smlal v1.8h, v7.8b, v27.8b  \n"
            "       sadalp v22.4s, v0.8h        \n"
            "       sadalp v23.4s, v1.8h        \n"

            "       add %0, %0, #32     \n"
            "       add %1, %1, #32     \n"
            "       subs w20, w20, #2   \n"
            "       bne 2b              \n"
            // start nd2
            "    3:                     \n"
            "        addp v8.4s, v8.4s, v9.4s   \n"
            "        addp v10.4s, v10.4s, v11.4s\n"
            "        addp v12.4s, v12.4s, v13.4s\n"
            "        addp v14.4s, v14.4s, v15.4s\n"
            "        addp v16.4s, v16.4s, v17.4s\n"
            "        addp v18.4s, v18.4s, v19.4s\n"
            "        addp v20.4s, v20.4s, v21.4s\n"
            "        addp v22.4s, v22.4s, v23.4s\n"

            "        addp v8.4s, v8.4s, v10.4s  \n"
            "        addp v9.4s, v12.4s, v14.4s \n"
            "        addp v10.4s, v16.4s, v18.4s\n"
            "        addp v11.4s, v20.4s, v22.4s\n"

            "        // start process kd4 kd2 kd1 cases\n"
            "    1:                     \n"
            "        cmp %w8, #0        \n"
            "        beq 4f             \n"
            "        // start subkernel_m4n4k4\n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  // load B4x4\n"
            "        sxtl v4.8h, v4.8b      \n"
            "        mov v6.d[0], v4.d[1]   \n"
            "        sxtl v5.8h, v5.8b      \n"
            "        mov v7.d[0], v5.d[1]   \n"

            "        ld1 {v2.8b}, [%0], #8  // load A2x4\n"
            "        sxtl v2.8h, v2.8b      \n"
            "        mov v3.d[0], v2.d[1]   \n"

            "        smull v12.4s, v2.4h, v4.4h \n"
            "        smull v13.4s, v2.4h, v6.4h \n"
            "        smull v14.4s, v2.4h, v5.4h \n"
            "        addp v12.4s, v12.4s, v13.4s\n"
            "        smull v15.4s, v2.4h, v7.4h \n"

            "        addp v14.4s, v14.4s, v15.4s\n"
            "        addp v12.4s, v12.4s, v14.4s\n"

            "        smull v16.4s, v3.4h, v4.4h \n"
            "        add v8.4s, v8.4s, v12.4s   \n"

            "        smull v17.4s, v3.4h, v6.4h \n"
            "        smull v18.4s, v3.4h, v5.4h \n"
            "        addp v16.4s, v16.4s, v17.4s\n"
            "        smull v19.4s, v3.4h, v7.4h \n"
            "        addp v18.4s, v18.4s, v19.4s\n"
            "        addp v16.4s, v16.4s, v18.4s\n"
            "        add v9.4s, v9.4s, v16.4s   \n"

            "        ld1 {v2.8b}, [%0], #8 // load next A2x4\n"
            "        sxtl v2.8h, v2.8b          \n"
            "        mov v3.d[0], v2.d[1]       \n"

            "        smull v12.4s, v2.4h, v4.4h \n"
            "        smull v13.4s, v2.4h, v6.4h \n"
            "        smull v14.4s, v2.4h, v5.4h \n"
            "        addp v12.4s, v12.4s, v13.4s\n"
            "        smull v15.4s, v2.4h, v7.4h \n"

            "        addp v14.4s, v14.4s, v15.4s\n"
            "        addp v12.4s, v12.4s, v14.4s\n"
            "        smull v16.4s, v3.4h, v4.4h \n"
            "        add v10.4s, v10.4s, v12.4s \n"

            "        smull v17.4s, v3.4h, v6.4h \n"
            "        smull v18.4s, v3.4h, v5.4h \n"
            "        addp v16.4s, v16.4s, v17.4s\n"
            "        smull v19.4s, v3.4h, v7.4h \n"

            "        addp v18.4s, v18.4s, v19.4s\n"
            "        addp v16.4s, v16.4s, v18.4s\n"
            "        add v11.4s, v11.4s, v16.4s \n"

            "    4:                             \n"
            "        cmp %w9, #0                \n"
            "        beq 5f                     \n"
            "        // start subkernel_m4n4k2  \n"
            "        ld1 {v0.8b}, [%1], #8   // load B2x4   \n"
            "                            // 00 11 22 33     \n"
            "        rev32 v1.4h, v0.4h  // 11 00 33 22     \n"
            "        rev64 v2.2s, v0.2s  // 22 33 00 11     \n"
            "        ld1 {v4.8b}, [%0], #8   // load A4x2   \n"
            "        rev64 v3.4h, v0.4h  // 33 22 11 00     \n"

            "        smull v12.8h, v4.8b, v0.8b \n"
            "        smull v13.8h, v4.8b, v1.8b \n"
            "        saddlp v12.4s, v12.8h      \n"
            "        smull v14.8h, v4.8b, v2.8b \n"
            "        saddlp v13.4s, v13.8h      \n"
            "        smull v15.8h, v4.8b, v3.8b \n"
            "        saddlp v14.4s, v14.8h      \n"
            "        saddlp v15.4s, v15.8h      \n"

            "        mov v16.s[0], v12.s[0] \n"
            "        mov v16.s[1], v13.s[0] \n"
            "        mov v16.s[2], v14.s[0] \n"
            "        mov v16.s[3], v15.s[0] \n"

            "        mov v17.s[0], v13.s[1] \n"
            "        mov v17.s[1], v12.s[1] \n"
            "        mov v17.s[2], v15.s[1] \n"
            "        mov v17.s[3], v14.s[1] \n"

            "        mov v18.s[0], v14.s[2] \n"
            "        mov v18.s[1], v15.s[2] \n"
            "        mov v18.s[2], v12.s[2] \n"
            "        mov v18.s[3], v13.s[2] \n"

            "        mov v19.s[0], v15.s[3] \n"
            "        mov v19.s[1], v14.s[3] \n"
            "        mov v19.s[2], v13.s[3] \n"
            "        mov v19.s[3], v12.s[3] \n"

            "        add v8.4s, v8.4s, v16.4s   \n"
            "        add v9.4s, v9.4s, v17.4s   \n"
            "        add v10.4s, v10.4s, v18.4s \n"
            "        add v11.4s, v11.4s, v19.4s \n"

            "    5:                         \n"
            "        cmp %w10, #0           \n"
            "        beq 6f                   \n"
            "        // start subkernel_m4n4k1\n"
            "        ld1 {v4.8b}, [%1]   // load B1x4\n"
            "        add %1, %1, #4     \n"
            "        ld1 {v2.8b}, [%0]   // load A4x1\n"
            "        add %0, %0, #4     \n"
            "        sxtl v4.8h, v4.8b  \n"
            "        sxtl v2.8h, v2.8b  \n"
            "        smlal v8.4s, v4.4h, v2.h[0]    \n"
            "        smlal v9.4s, v4.4h, v2.h[1]    \n"
            "        smlal v10.4s, v4.4h, v2.h[2]   \n"
            "        smlal v11.4s, v4.4h, v2.h[3]   \n"

            "    6:                            \n"
            "        cmp %12, #0               \n"
            "        beq 9f                    \n"

            "        ld1 {v12.4s}, [%12] \n"
            "        // int32 => fp32       \n"
            "        scvtf v8.4s, v8.4s     \n"
            "        scvtf v9.4s, v9.4s     \n"
            "        scvtf v10.4s, v10.4s   \n"
            "        scvtf v11.4s, v11.4s   \n"
            "        // fp32 *= scale_tm    \n"
            "        fmul v8.4s, v8.4s, v12.s[0]    \n"
            "        fmul v9.4s, v9.4s, v12.s[1]    \n"
            "        fmul v10.4s, v10.4s, v12.s[2]  \n"
            "        fmul v11.4s, v11.4s, v12.s[3]  \n"

            "        cmp %13, #0            \n"
            "        beq 7f                 \n"

            "        ld1 {v14.4s}, [%13]    \n"
            "        dup v15.4s, v14.s[0]   \n"
            "        fadd v8.4s, v8.4s, v15.4s  \n"
            "        dup v15.4s, v14.s[1]       \n"
            "        fadd v9.4s, v9.4s, v15.4s  \n"
            "        dup v15.4s, v14.s[2]       \n"
            "        fadd v10.4s, v10.4s, v15.4s\n"
            "        dup v15.4s, v14.s[3]       \n"
            "        fadd v11.4s, v11.4s, v15.4s\n"

            "        7:                         \n"
            "            // fp32 -> int32       \n"
            "            fcvtas v8.4s, v8.4s    \n"
            "            fcvtas v9.4s, v9.4s    \n"
            "            fcvtas v10.4s, v10.4s  \n"
            "            fcvtas v11.4s, v11.4s  \n"
            "            // int32 -> int16      \n"
            "            sqxtn v6.4h, v8.4s     \n"
            "            sqxtn2 v6.8h, v9.4s    \n"
            "            sqxtn v7.4h, v10.4s    \n"
            "            sqxtn2 v7.8h, v11.4s   \n"
            "            // int16 -> int8       \n"
            "            sqxtn v8.8b, v6.8h     \n"
            "            sqxtn v9.8b, v7.8h     \n"
            "            // save                \n"
            "            st1 {v8.s}[0], [%2]    \n"
            "            add %x2, %x2, #4       \n"
            "            st1 {v8.s}[1], [%3]    \n"
            "            add %x3, %x3, #4       \n"
            "            st1 {v9.s}[0], [%4]    \n"
            "            add %x4, %x4, #4       \n"
            "            st1 {v9.s}[1], [%5]    \n"
            "            add %x5, %x5, #4       \n"
            "            b 10f   \n"

            "    9:                             \n"
            "        st1 {v8.4s}, [%x2], #16    \n"
            "        st1 {v9.4s}, [%x3], #16    \n"
            "        st1 {v10.4s}, [%x4], #16   \n"
            "        st1 {v11.4s}, [%x5], #16   \n"

            "    10:         \n"
            "        subs %x11, %x11, #1     \n"
            "        mov %x0, x8             \n"
            "        bne 8b                  \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc0),     // %2
            "=r"(pc1),     // %3
            "=r"(pc2),     // %4
            "=r"(pc3),     // %5
            "=r"(k8_even), // %6
            "=r"(k8),      // %7
            "=r"(k4),      // %8
            "=r"(k2),      // %9
            "=r"(k1),      // %10
            "=r"(n4),      // %11
            "=r"(scales),  // %12
            "=r"(bias)     // %13
            : "0"(pa),
            "1"(pb),
            "2"(pc0),
            "3"(pc1),
            "4"(pc2),
            "5"(pc3),
            "6"(k8_even),
            "7"(k8),
            "8"(k4),
            "9"(k2),
            "10"(k1),
            "11"(n4),
            "12"(scales),
            "13"(bias)
            : "cc", "memory", "x8", "w20", "x14", "x15", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }

    if (n2 > 0)
    {
        asm volatile(
            "    eor v8.8b, v8.8b, v8.8b    \n"
            "    eor v9.8b, v9.8b, v9.8b    \n"
            "    eor v10.8b, v10.8b, v10.8b \n"
            "    eor v11.8b, v11.8b, v11.8b \n"
            "    eor v12.8b, v12.8b, v12.8b \n"
            "    eor v13.8b, v13.8b, v13.8b \n"
            "    eor v14.8b, v14.8b, v14.8b \n"
            "    eor v15.8b, v15.8b, v15.8b \n"

            "    eor v16.8b, v16.8b, v16.8b \n"
            "    eor v17.8b, v17.8b, v17.8b \n"
            "    eor v18.8b, v18.8b, v18.8b \n"
            "    eor v19.8b, v19.8b, v19.8b \n"
            "    eor v20.8b, v20.8b, v20.8b \n"
            "    eor v21.8b, v21.8b, v21.8b \n"
            "    eor v22.8b, v22.8b, v22.8b \n"
            "    eor v23.8b, v23.8b, v23.8b \n"

            "9:                \n"
            "    mov x8, %x0  // PanelA     \n"

            "    cmp %w7, #0                \n"
            "    beq 1f         // k <= 7   \n"

            "    mov w20, %w7                \n"
            "    cmp %w6, #0                 \n"
            "    beq 2f// loop number is even \n"

            "    // start loopkd8_nd2       \n"
            "    subs w20, w20, #1          \n"
            "    ld1 {v4.8b, v5.8b}, [%1], #16  // load two lines of B\n"
            "    ld1 {v2.8b, v3.8b}, [%0], #16  // load two lines of PanelA\n"
            "    smull v0.8h, v4.8b, v2.8b  \n"
            "    smull v1.8h, v4.8b, v3.8b  \n"
            "    saddlp v8.4s, v0.8h        \n"
            "    saddlp v12.4s, v1.8h       \n"
            "    smull v0.8h, v5.8b, v2.8b  \n"
            "    smull v1.8h, v5.8b, v3.8b  \n"
            "    saddlp v9.4s, v0.8h        \n"
            "    saddlp v13.4s, v1.8h       \n"

            "    ld1 {v2.8b, v3.8b}, [%0], #16  \n"
            "    smull v0.8h, v4.8b, v2.8b      \n"
            "    smull v1.8h, v4.8b, v3.8b      \n"
            "    saddlp v16.4s, v0.8h       \n"
            "    saddlp v20.4s, v1.8h       \n"
            "    smull v0.8h, v5.8b, v2.8b  \n"
            "    smull v1.8h, v5.8b, v3.8b  \n"
            "    saddlp v17.4s, v0.8h       \n"
            "    saddlp v21.4s, v1.8h       \n"

            "    cmp w20, #0                \n"
            "    beq 3f                     \n"

            "    2: \n"
            "        add x15, %1, #16 \n"
            "        add x14, %0, #32 \n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  \n"
            "        ld1 {v2.8b, v3.8b}, [%0], #16  \n"

            "        smull v0.8h, v4.8b, v2.8b      \n"
            "        ld1 {v6.8b, v7.8b}, [x15], #16 \n"
            "        smull v1.8h, v5.8b, v2.8b        \n"
            "        ld1 {v24.8b, v25.8b}, [x14], #16 \n"

            "        smlal v0.8h, v6.8b, v24.8b     \n"
            "        smlal v1.8h, v7.8b, v24.8b     \n"

            "        sadalp v8.4s, v0.8h    \n"
            "        sadalp v9.4s, v1.8h    \n"

            "        smull v0.8h, v4.8b, v3.8b  \n"
            "        smull v1.8h, v5.8b, v3.8b  \n"
            "        smlal v0.8h, v6.8b, v25.8b \n"
            "        smlal v1.8h, v7.8b, v25.8b \n"

            "        sadalp v12.4s, v0.8h   \n"
            "        sadalp v13.4s, v1.8h   \n"

            "        // finish v8v9 v12v13, start proc v16v17,v20v21\n"
            "        ld1 {v28.8b, v29.8b}, [%0], #16\n"
            "        smull v0.8h, v4.8b, v28.8b\n"
            "        smull v1.8h, v5.8b, v28.8b\n"
            "        ld1 {v26.8b, v27.8b}, [x14], #16\n"
            "        smlal v0.8h, v6.8b, v26.8b\n"
            "        smlal v1.8h, v7.8b, v26.8b\n"
            "        sadalp v16.4s, v0.8h\n"
            "        sadalp v17.4s, v1.8h\n"

            "        smull v0.8h, v4.8b, v29.8b\n"
            "        smull v1.8h, v5.8b, v29.8b\n"
            "        smlal v0.8h, v6.8b, v27.8b\n"
            "        smlal v1.8h, v7.8b, v27.8b\n"
            "        sadalp v20.4s, v0.8h\n"
            "        sadalp v21.4s, v1.8h\n"

            "        add %0, %0, #32        \n"
            "        add %1, %1, #16        \n"
            "        subs w20, w20, #2      \n"
            "        bne 2b                 \n"

            "    3:                             \n"
            "        addp v8.4s, v8.4s, v9.4s   \n"
            "        addp v12.4s, v12.4s, v13.4s\n"
            "        addp v16.4s, v16.4s, v17.4s\n"
            "        addp v20.4s, v20.4s, v21.4s\n"

            "        addp v8.4s, v8.4s, v8.4s   \n"
            "        addp v12.4s, v12.4s, v12.4s\n"
            "        addp v16.4s, v16.4s, v16.4s\n"
            "        addp v20.4s, v20.4s, v20.4s\n"

            "        // start process kd4 kd2 kd1 cases\n"
            "    1:   \n"
            "        cmp %w8, 0 \n"
            "        beq 4f     \n"
            "        // start subkernel_m4n2k4  \n"
            "        ld1 {v4.8b}, [%1], #8  // load B4x2\n"
            "        sxtl v4.8h, v4.8b      \n"
            "        mov v6.d[0], v4.d[1]   \n"

            "        ld1 {v2.8b}, [%0], #8  // load first A2x4\n"
            "        sxtl v2.8h, v2.8b      \n"
            "        mov v3.d[0], v2.d[1]   \n"

            "        smull v9.4s, v2.4h, v4.4h  \n"
            "        smull v10.4s, v2.4h, v6.4h \n"

            "        addp v9.4s, v9.4s, v10.4s  \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        add v8.4s, v8.4s, v9.4s    \n"

            "        smull v13.4s, v3.4h, v4.4h \n"
            "        smull v14.4s, v3.4h, v6.4h \n"

            "        addp v13.4s, v13.4s, v14.4s\n"
            "        addp v13.4s, v13.4s, v13.4s\n"
            "        add v12.4s, v12.4s, v13.4s \n"

            "        ld1 {v2.8b}, [%0], #8 // load next A2x4\n"
            "        sxtl v2.8h, v2.8b          \n"
            "        mov v3.d[0], v2.d[1]       \n"

            "        smull v17.4s, v2.4h, v4.4h \n"
            "        smull v18.4s, v2.4h, v6.4h \n"

            "        addp v17.4s, v17.4s, v18.4s\n"
            "        addp v17.4s, v17.4s, v17.4s\n"
            "        add v16.4s, v16.4s, v17.4s \n"

            "        smull v21.4s, v3.4h, v4.4h \n"
            "        smull v22.4s, v3.4h, v6.4h \n"

            "        addp v21.4s, v21.4s, v22.4s\n"
            "        addp v21.4s, v21.4s, v21.4s\n"
            "        add v20.4s, v20.4s, v21.4s \n"

            "    4:                             \n"
            "        cmp %w9, 0                 \n"
            "        beq 5f                     \n"
            "        // start subkernel_m4n2k2  \n"
            "        ld1 {v4.8b}, [%0], #8   //load A4x2\n"
            "        ld1 {v0.8b}, [%1]   // load B2x2   \n"
            "        add %1, %1, #4                     \n"
            "                            // 00 11 22 33 \n"
            "        rev32 v1.4h, v0.4h  // 11 00 33 22 \n"
            "        rev64 v2.2s, v0.2s  // 22 33 00 11 \n"
            "        rev64 v3.4h, v0.4h  // 33 22 11 00 \n"

            "        smull v21.8h, v4.8b, v0.8b \n"
            "        smull v22.8h, v4.8b, v1.8b \n"
            "        smull v23.8h, v4.8b, v2.8b \n"
            "        smull v24.8h, v4.8b, v3.8b \n"

            "        saddlp v21.4s, v21.8h  \n"
            "        saddlp v22.4s, v22.8h  \n"
            "        saddlp v23.4s, v23.8h  \n"
            "        saddlp v24.4s, v24.8h  \n"

            "        mov v9.s[0], v21.s[0]  \n"
            "        mov v9.s[1], v22.s[0]  \n"
            "        add v8.4s, v8.4s, v9.4s\n"

            "        mov v13.s[0], v22.s[1] \n"
            "        mov v13.s[1], v21.s[1] \n"
            "        add v12.4s, v12.4s, v13.4s \n"

            "        mov v17.s[0], v23.s[2] \n"
            "        mov v17.s[1], v24.s[2] \n"
            "        add v16.4s, v16.4s, v17.4s \n"

            "        mov v21.s[0], v24.s[3]     \n"
            "        mov v21.s[1], v23.s[3]     \n"
            "        add v20.4s, v20.4s, v21.4s \n"

            "    5:   \n"
            "        cmp %w10, 0    \n"
            "        beq 6f         \n"
            "        // start subkernel_m4n2k1\n"
            "        ld1 {v4.8b}, [%1]   // load B1x2\n"
            "        add %1, %1, #2     \n"
            "        ld1 {v2.8b}, [%0]   // load A4x1\n"
            "        add %0, %0, #4     \n"
            "        sxtl v4.8h, v4.8b  \n"
            "        sxtl v2.8h, v2.8b  \n"
            "        smlal v8.4s, v4.4h, v2.h[0]    \n"
            "        smlal v12.4s, v4.4h, v2.h[1]   \n"
            "        smlal v16.4s, v4.4h, v2.h[2]   \n"
            "        smlal v20.4s, v4.4h, v2.h[3]   \n"

            "    6:               \n"
            "        cmp %11, #0            \n"
            "        beq 7f                 \n"

            "        mov v8.d[1], v12.d[0]  \n"
            "        mov v16.d[1], v20.d[0] \n"

            "        // v12: 0 1 2 3        \n"
            "        ld1 {v12.4s}, [%11]     \n"
            "        zip2 v13.4s, v12.4s, v12.4s    \n"
            "        zip1 v12.4s, v12.4s, v12.4s    \n"
            "        // v12: 0 0 1 1        \n"
            "        // v13: 2 2 3 3        \n"

            "        // int32 => fp32   \n"
            "        scvtf v8.4s, v8.4s \n"
            "        scvtf v16.4s, v16.4s   \n"
            "        // fp32 *= scale_tm    \n"
            "        fmul v8.4s, v8.4s, v12.4s  \n"
            "        fmul v16.4s, v16.4s, v13.4s\n"

            "        cmp %12, #0   \n"
            "        beq 8f // skip add scales  \n"

            "        // fp32 += scales_tm       \n"
            "        ld1 {v12.4s}, [%12]        \n"
            "        zip2 v13.4s, v12.4s, v12.4s\n"
            "        zip1 v12.4s, v12.4s, v12.4s\n"
            "        fadd v8.4s, v8.4s, v12.4s  \n"
            "        fadd v16.4s, v16.4s, v13.4s\n"

            "        8:                         \n"
            "            // fp32 -> int32       \n"
            "            fcvtas v8.4s, v8.4s    \n"
            "            fcvtas v16.4s, v16.4s  \n"
            "            // int32 -> int16      \n"
            "            sqxtn v8.4h, v8.4s     \n"
            "            sqxtn v16.4h, v16.4s   \n"
            "            // int16 -> int8       \n"
            "            sqxtn v8.8b, v8.8h     \n"
            "            sqxtn v16.8b, v16.8h   \n"
            "            // save                \n"
            "            st1 {v8.h}[0], [%2]    \n"
            "            add %2, %2, #2         \n"
            "            st1 {v8.h}[1], [%3]    \n"
            "            add %3, %3, #2         \n"
            "            st1 {v16.h}[0], [%4]   \n"
            "            add %4, %4, #2         \n"
            "            st1 {v16.h}[1], [%5]   \n"
            "            add %5, %5, #2         \n"
            "            b 10f   \n"

            "    7:             \n"
            "        st1 {v8.2s}, [%2], #8     \n"
            "        st1 {v12.2s}, [%3], #8    \n"
            "        st1 {v16.2s}, [%4], #8    \n"
            "        st1 {v20.2s}, [%5], #8    \n"

            "    10:            \n"
            "        mov %0, x8                 \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc0),     // %2
            "=r"(pc1),     // %3
            "=r"(pc2),     // %4
            "=r"(pc3),     // %5
            "=r"(k8_even), // %6
            "=r"(k8),      // %7
            "=r"(k4),      // %8
            "=r"(k2),      // %9
            "=r"(k1),      // %10
            "=r"(scales),  // %11
            "=r"(bias)     // %12
            : "0"(pa),
            "1"(pb),
            "2"(pc0),
            "3"(pc1),
            "4"(pc2),
            "5"(pc3),
            "6"(k8_even),
            "7"(k8),
            "8"(k4),
            "9"(k2),
            "10"(k1),
            "11"(scales),
            "12"(bias)
            : "cc", "memory", "x8", "w20", "x14", "x15", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }

    if (n1 > 0)
    {
        asm volatile(
            "    eor v8.8b, v8.8b, v8.8b    \n"
            "    eor v9.8b, v9.8b, v9.8b    \n"
            "    eor v10.8b, v10.8b, v10.8b \n"
            "    eor v11.8b, v11.8b, v11.8b \n"
            "    eor v12.8b, v12.8b, v12.8b \n"
            "    eor v13.8b, v13.8b, v13.8b \n"
            "    eor v14.8b, v14.8b, v14.8b \n"
            "    eor v15.8b, v15.8b, v15.8b \n"

            "    eor v16.8b, v16.8b, v16.8b \n"
            "    eor v17.8b, v17.8b, v17.8b \n"
            "    eor v18.8b, v18.8b, v18.8b \n"
            "    eor v19.8b, v19.8b, v19.8b \n"
            "    eor v20.8b, v20.8b, v20.8b \n"
            "    eor v21.8b, v21.8b, v21.8b \n"
            "    eor v22.8b, v22.8b, v22.8b \n"
            "    eor v23.8b, v23.8b, v23.8b \n"

            "1:                      \n"
            "    cmp %w7, #0                     \n"
            "    beq 10f             \n"

            "    mov w20, %w7        \n"
            "    cmp %w6, #0         \n"
            "    beq 11f// loop number is even \n"

            "    // start loopkd8_nd1       \n"
            "    subs w20, w20, #1          \n"
            "    ld1 {v4.8b}, [%1], #8  // load four lines of B\n"
            "    ld1 {v2.8b, v3.8b}, [%0], #16  // load two lines of PanelA\n"
            "    smull v0.8h, v4.8b, v2.8b  \n"
            "    smull v1.8h, v4.8b, v3.8b  \n"
            "    saddlp v8.4s, v0.8h        \n"
            "    saddlp v12.4s, v1.8h       \n"

            "    ld1 {v2.8b, v3.8b}, [%0], #16  \n"
            "    smull v0.8h, v4.8b, v2.8b  \n"
            "    smull v1.8h, v4.8b, v3.8b  \n"
            "    saddlp v16.4s, v0.8h       \n"
            "    saddlp v20.4s, v1.8h       \n"

            "    cmp w20, #0                \n"
            "    beq 12f                    \n"

            "    11:          \n"
            "        ld1 {v4.8b, v5.8b}, [%1], #16  \n"
            "        ld1 {v24.8b, v25.8b, v26.8b, v27.8b}, [%0], #32\n"
            "        ld1 {v28.8b, v29.8b, v30.8b, v31.8b}, [%0], #32\n"

            "        smull v0.8h, v24.8b, v4.8b \n"
            "        smlal v0.8h, v28.8b, v5.8b \n"
            "        sadalp v8.4s, v0.8h        \n"

            "        smull v1.8h, v25.8b, v4.8b \n"
            "        smlal v1.8h, v29.8b, v5.8b \n"
            "        sadalp v12.4s, v1.8h       \n"

            "        smull v0.8h, v26.8b, v4.8b \n"
            "        smlal v0.8h, v30.8b, v5.8b \n"
            "        sadalp v16.4s, v0.8h       \n"

            "        smull v1.8h, v27.8b, v4.8b \n"
            "        smlal v1.8h, v31.8b, v5.8b \n"
            "        sadalp v20.4s, v1.8h       \n"

            "        subs w20, w20, #2          \n"
            "        bne 11b                    \n"

            "    12:                            \n"
            "        addp v8.4s, v8.4s, v8.4s   \n"
            "        addp v8.4s, v8.4s, v8.4s   \n"
            "        addp v12.4s, v12.4s, v12.4s\n"
            "        addp v12.4s, v12.4s, v12.4s\n"
            "        addp v16.4s, v16.4s, v16.4s\n"
            "        addp v16.4s, v16.4s, v16.4s\n"
            "        addp v20.4s, v20.4s, v20.4s\n"
            "        addp v20.4s, v20.4s, v20.4s\n"

            "        // start process kd4 kd2 kd1 cases\n"
            "    10:                        \n"
            "        cmp %w8, #0            \n"
            "        beq 13f                        \n"
            "        // start subkernel_m4n1k2      \n"
            "        ld1 {v4.8b}, [%1]  // load B4x1\n"
            "        add %x1, %x1, #4       \n"
            "        sxtl v4.8h, v4.8b   // extend B4x1 to v4   \n"

            "        ld1 {v2.8b, v3.8b}, [%0], #16  // load A4x4\n"
            "        sxtl v2.8h, v2.8b      \n"
            "        mov v5.d[0], v2.d[1]   \n"
            "        sxtl v3.8h, v3.8b      \n"
            "        mov v6.d[0], v3.d[1]  // extend A4x4 to v2,v5,v3,v6\n"

            "        smull v9.4s, v2.4h, v4.4h  \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        addp v9.4s, v9.4s, v9.4s   \n"
            "        add v8.4s, v8.4s, v9.4s    \n"

            "        smull v13.4s, v5.4h, v4.4h \n"
            "        addp v13.4s, v13.4s, v13.4s\n"
            "        addp v13.4s, v13.4s, v13.4s\n"
            "        add v12.4s, v12.4s, v13.4s \n"

            "        smull v17.4s, v3.4h, v4.4h \n"
            "        addp v17.4s, v17.4s, v17.4s\n"
            "        addp v17.4s, v17.4s, v17.4s\n"
            "        add v16.4s, v16.4s, v17.4s \n"

            "        smull v21.4s, v6.4h, v4.4h \n"
            "        addp v21.4s, v21.4s, v21.4s\n"
            "        addp v21.4s, v21.4s, v21.4s\n"
            "        add v20.4s, v20.4s, v21.4s \n"

            "    13:                            \n"
            "        cmp %w9, #0                \n"
            "        beq 14f                    \n"
            "        // start subkernel_m4n1k2  \n"
            "        ld1 {v4.8b}, [%0], #8   // load A4x2   \n"
            "        ld1 {v0.8b}, [%1]   // load B2x1   \n"
            "        add %1, %1, #2             \n"

            "        mov v0.h[1], v0.h[0]   \n"
            "        mov v0.s[1], v0.s[0]   \n"

            "        smull v0.8h, v0.8b, v4.8b  \n"
            "        saddlp v0.4s, v0.8h        \n"

            "        mov v9.s[0], v0.s[0]       \n"
            "        add v8.4s, v8.4s, v9.4s    \n"
            "        mov v13.s[0], v0.s[1]      \n"
            "        add v12.4s, v12.4s, v13.4s \n"
            "        mov v17.s[0], v0.s[2]      \n"
            "        add v16.4s, v16.4s, v17.4s \n"
            "        mov v21.s[0], v0.s[3]      \n"
            "        add v20.4s, v20.4s, v21.4s \n"

            "    14:                            \n"
            "        cmp %w10, #0               \n"
            "        beq 15f                    \n"
            "        // start subkernel_m4n1k1  \n"
            "        ld1 {v4.8b}, [%1]   // load B1x1\n"

            "        add %1, %1, #1             \n"
            "        ld1 {v2.8b}, [%0]   // load A4x1\n"
            "        add %0, %0, #4             \n"
            "        sxtl v4.8h, v4.8b          \n"
            "        sxtl v2.8h, v2.8b          \n"

            "        smull v0.4s, v2.4h, v4.h[0]\n"

            "        add v8.4s, v8.4s, v0.4s    \n"

            "        mov v13.s[0], v0.s[1]      \n"
            "        add v12.4s, v12.4s, v13.4s \n"

            "        mov v17.s[0], v0.s[2]      \n"
            "        add v16.4s, v16.4s, v17.4s \n"

            "        mov v21.s[0], v0.s[3]      \n"
            "        add v20.4s, v20.4s, v21.4s \n"

            "    15:                           \n"
            // REQUANT
            "        cmp %11, #0            \n"
            "        beq 16f                \n"

            "        mov v8.s[1], v12.s[0]  \n"
            "        mov v8.s[2], v16.s[0]  \n"
            "        mov v8.s[3], v20.s[0]  \n"

            "        // v12: s0 s1 s2 s3    \n"
            "        ld1 {v12.4s}, [%11] \n"
            "        // int32 => fp32       \n"
            "        scvtf v8.4s, v8.4s     \n"
            "        // fp32 *= scale_tm    \n"
            "        fmul v8.4s, v8.4s, v12.4s  \n"

            "        cmp %12, #0                \n"
            "        beq 17f                    \n"

            "        // fp32 += bias_tm         \n"
            "        ld1 {v12.4s}, [%12]        \n"
            "        fadd v8.4s, v8.4s, v12.4s  \n"

            "        17:                        \n"
            "            // fp32 -> int32       \n"
            "            fcvtas v8.4s, v8.4s    \n"
            "            // int32 -> int16      \n"
            "            sqxtn v8.4h, v8.4s     \n"
            "            // int16 -> int8       \n"
            "            sqxtn v8.8b, v8.8h     \n"
            "            // save                \n"
            "            st1 {v8.b}[0], [%2]   \n"
            "            st1 {v8.b}[1], [%3]   \n"
            "            st1 {v8.b}[2], [%4]   \n"
            "            st1 {v8.b}[3], [%5]   \n"
            "            b 2f          \n"

            "        // no need to add the last output pointer\n"
            "    16:                           \n"
            "        st1 {v8.s}[0], [%2]       \n"
            "        st1 {v12.s}[0], [%3]      \n"
            "        st1 {v16.s}[0], [%4]      \n"
            "        st1 {v20.s}[0], [%5]      \n"

            "    2:                            \n"
            "        mov x0, #0                 \n"
            : "=r"(pa),      // %0
            "=r"(pb),      // %1
            "=r"(pc0),     // %2
            "=r"(pc1),     // %3
            "=r"(pc2),     // %4
            "=r"(pc3),     // %5
            "=r"(k8_even), // %6
            "=r"(k8),      // %7
            "=r"(k4),      // %8
            "=r"(k2),      // %9
            "=r"(k1),      // %10
            "=r"(scales),  // %11
            "=r"(bias)     // %12
            : "0"(pa),
            "1"(pb),
            "2"(pc0),
            "3"(pc1),
            "4"(pc2),
            "5"(pc3),
            "6"(k8_even),
            "7"(k8),
            "8"(k4),
            "9"(k2),
            "10"(k1),
            "11"(scales),
            "12"(bias)
            : "cc", "memory", "x0", "x8", "w20", "x14", "x15", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }
}
#undef DECOMPOSE_K
#undef DECOMPOSE_N

static void int8kernel(void* dst, const int8_t* sa, const int8_t* sb, int m, int k, int n, int ldc, float* scales, float* bias)
{
    int8_t* pa = (int8_t*)sa;
    int8_t* pb = (int8_t*)sb;
    const int nn = (m >> 2) << 2;
    if (scales == 0)
    {
        int32_t* pc = (int32_t*)dst;
#if PRINT_MATRIX
        int32_t* origin = pc;
#endif

        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < nn; i += 4)
        {
            int8kernel_m4((void*)(pc + i * ldc), pa + i * k, pb, m, k, n, ldc, 0, 0);
        }

        pa += nn * k;
        pc += nn * ldc;

        switch (m - nn)
        {
        case 3:
            int8kernel_m2((void*)pc, pa, pb, m, k, n, ldc, 0, 0);
            pc += 2 * ldc;
            pa += 2 * k;
            int8kernel_m1((void*)pc, pa, pb, m, k, n, ldc, 0, 0);
            break;
        case 2:
            int8kernel_m2((void*)pc, pa, pb, m, k, n, ldc, 0, 0);
            break;
        case 1:
            int8kernel_m1((void*)pc, pa, pb, m, k, n, ldc, 0, 0);
            break;
        case 0:
        default:
            break;
        }
#if PRINT_MATRIX
        print_int32_matrix("pc", origin, m, n, ldc);
#endif
    }
    else
    {
        int8_t* pc = (int8_t*)dst;

#if PRINT_MATRIX
        print_fp32_vec("scales", scales, m);
#endif
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < nn; i += 4)
        {
            int8kernel_m4((void*)(pc + i * ldc), pa + i * k, pb, m, k, n, ldc, scales + i, (bias == 0) ? 0 : bias + i);
        }

        pa += nn * k;
        pc += nn * ldc;

        scales += nn;
        bias = (bias == 0) ? 0 : bias + nn;

        switch (m - nn)
        {
        case 3:
            int8kernel_m2((void*)pc, pa, pb, m, k, n, ldc, scales, bias);
            pc += 2 * ldc;
            pa += 2 * k;
            scales += 2;
            bias = (bias == 0) ? 0 : bias + 2;
            int8kernel_m1((void*)pc, pa, pb, m, k, n, ldc, scales, bias);
            break;
        case 2:
            int8kernel_m2((void*)pc, pa, pb, m, k, n, ldc, scales, bias);
            break;
        case 1:
            int8kernel_m1((void*)pc, pa, pb, m, k, n, ldc, scales, bias);
            break;
        case 0:
        default:
            break;
        }
    }
    return;
}


// input nchw, output nchw
void fwdConvolution_nchw_mm(int8_t *input, int8_t *kernel, int8_t *bias, int8_t *output, 
                      float inS, float kS, float bS, float outS,
                      int32_t inOffset, int32_t filterOffset, int32_t biasOffset, int32_t outOffset,
                      dim_t N, dim_t C, dim_t H, dim_t W, dim_t KN, dim_t KH, dim_t KW, int pad_size, int stride_size, size_t group,
                      size_t dilation, bool doRelu, bool doBias, 
                      int out_h, int out_w) {

  // only support KN%4
  assert(KN%4 ==0);
  assert((out_w*out_w)%16==0);

  
  float outScale = outS;
  float inScale = inS;
  float filterScale = kS;
  float biasScale = bS;

  // Calculate the scale of the values that come out of the matrix
  // multiplication part of the calculation.
    float matMulScale = inScale * filterScale;
    float scale = matMulScale / outScale;
    float bscale = biasScale / matMulScale;  


//#if DEBUG_INFO
#if 1
  printf("off i:%d f:%d b:%d o:%d\n",inOffset,  filterOffset,  biasOffset,  outOffset);

  printf("\nscale:%f\n", scale);
  printf("bscale:%f\n",bscale);
#endif


  int8_t* flat_input = (int8_t*)malloc (KH*KW*C* out_h * out_w); // c= channel
//  int8_t* kernel_vec = (int8_t*)malloc (KH*KW*C); // 1 filter * channel
  int32_t* mm_out_c = (int32_t*)malloc(C*KN*out_w*out_h*sizeof(int32_t)); // 채널별로 나눠서 저장 후 나중에 합침.
  
  
  
  // make patched input
  // For each convolution 'jump' in the input tensor:
  clock_t t1s = clock();
  
  

 
/*
   ssize_t x = -ssize_t(pad_size);
   for (dim_t ax = 0; ax < out_h; ax++, x += stride_size) {  //H
    ssize_t y = -ssize_t(pad_size);
    for (dim_t ay = 0; ay < out_w; y += stride_size, ay++) { 
       linearize_volume_nchw(input, flat_input + (ax * KH * KW * out_w * C) + (ay * KH * KW * C), x, y, KW, KH, KN, W, H, 1, W, W * H, pad_size > 0 ? 1 : 0, 0, 1, 1, inOffset);
    
#if 1
      if(ax<4 && ay <4) 
      {
        for(int c=0;c<3;c++)
        {
        printf("\n(%d,%d - %d)[%d,%d]\n",x,y,c,KW,KH);
        //ch0�� ���. 
        for(int k=0;k<KW*KH;k++) { printf("%02x ",flat_input[k + (ax * KH * KW * out_w * C) + (ay * KH * KW * C)+ c*KH*KW]); if(k%7==6) printf("/ "); }

        }
      }
#endif      
      
    }
  }

  const int stride = KH * KW * out_w * out_h;
  signed char* ret = (signed char*)flat_input;
  #pragma omp parallel for num_threads(4)
  for (int p = 0; p < C; p++)
  {
      const signed char* in = input + H*W*p;  // p channel
      int retID = stride * p;
      for (int u = 0; u < KH; u++)
      {
          for (int v = 0; v < KW; v++)
          {
              for (int i = 0; i < out_h; i++)
              {
                  for (int j = 0; j < out_w; j++)
                  {
                      int row = u + i * stride_size - pad_size;    
                      int col = v + j * stride_size - pad_size;
                      if(row<0 || col<0 || row >= H || col >= W) ret[retID]=0;
                      else {
                        int index = row * W + col;
                        ret[retID] = input[index];

                      }
                      retID++;
                  }
              }
          }
      }
  }
*/

  const int stride_in_c = W*H;
  const int stride_out_c = KH * KW * out_w * out_h;
  signed char* ret = (signed char*)flat_input;
 
  #pragma omp parallel for num_threads(3)
  for (int p = 0; p < C; p++)
  {
    int8_t *in = input + stride_in_c * p;
    int retID = stride_out_c * p;
  for (int i = 0; i < out_h; i++)
  {
    for (int j = 0; j < out_w; j++)
    {      

        for (int u = 0; u < KH; u++)
            {
                for (int v = 0; v < KW; v++)
                {            
                      int row = u + i * stride_size - pad_size;    
                      int col = v + j * stride_size - pad_size;
                      if(row<0 || col<0 || row >= H || col >= W) ret[retID]=0;
                      else {
                        int index = row * W + col;
                        ret[retID] = in[index];
                      }
                      retID++;
                  }
              }
          }
      }
  }


#if 0
// 위에는 kchw인데 col-major 로 되어있음
for(int ay=0, x=-pad_size;ay<4;ay++,x+=stride_size) {
  for(int ax=0, y=-pad_size;ax<4;ax++,y+=stride_size) {
        for(int c=0;c<3;c++)
        {
       printf("\ncc(%d,%d - %d)[%d,%d]\n",x,y,c,KW,KH);
        //ch0�� ���. 
        for(int k=0;k<KW*KH;k++) { printf("%02x ",flat_input[k + (ax * KH * KW) + (ay * KH * KW * out_w)+ c*KH*KW*out_w*out_h]); if(k%7==6) printf("/ "); }
        }

  }

}


#endif   

  clock_t t1e = clock();
  cout <<"linearize: : "<<(double)(t1e- t1s)/CLOCKS_PER_SEC*1000 << std::endl;

  const int m = KN;
    const int n = out_w * out_h;
    const int k = C * KW * KH;


  if(filterOffset!=0) printf("!!!!! check filterOffset");
  clock_t t2s = clock();
/*
    // GEMM
    int32_t* pc = mm_out_c;
    const int8_t* pa = kernel;
    int8_t* pb = flat_input;
    const size_t ldc = out_w*out_h; //top_blob.cstep;

    
    int8kernel((void*)pc, pa, pb, m, k, n, ldc, 0, 0);

  #pragma omp parallel for num_threads(4)
  for (dim_t d = 0; d < KN; d++) {
      vector_scale_bias_offset_clip_s8(mm_out_c + d*out_w*out_h, *((int32_t*)bias+d), output  + (d*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
  }

*/

  // For each output channel in the group:
  // row=4

  #pragma omp parallel for num_threads(4)
  for(int c=0;c<C;c++) {
    for (dim_t d = 0; d < KN; d+=4) { 
       
      mm_matrix_multiply_s8_v2(kernel + KH * KW * d + c*KH*KW*KN, flat_input + c*out_w*out_h*KW*KH, (int32_t *)mm_out_c + c*out_w*out_h, KH * KW, out_h * out_w, KH * KW, out_w, filterOffset);
      //mm_matrix_multiply_s8(kernel + KH * KW * d + c*KH*KW*KN, flat_input + c*out_w*out_h*KW*KH, (int32_t *)mm_out_c + c*out_w*out_h, KH * KW, out_h * out_w, KH * KW, filterOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + d *out_w*out_h, *((int32_t*)bias+d), output  + (d*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+1)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+1)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+2)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+2)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+3)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+3)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
  
    }
    
/*
  #pragma omp parallel for num_threads(3)
  for(int c=0;c<C;c++) {
    for (dim_t d = 0; d < KN; d++) { //8=k_row
       
      mm_matrix_multiply_s8(kernel + KH * KW * d + c*KH*KW*KN, flat_input + c*out_w*out_h*KW*KH, (int32_t *)mm_out_c + c*out_w*out_h, KH * KW, out_h * out_w, KH * KW, filterOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + d *out_w*out_h, *((int32_t*)bias+d), output  + (d*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+1)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+1)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+2)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+2)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+3)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+3)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
  
    }    
*/


#if DEBUG_INFO
      printf("kernel %d\n",d);
      for(int c=0;c<C;c++)
      {
        for (int j = 0; j < KH; j++)
        {
          for (int i = 0; i < KW; i++)
            {
                printf("%x ",kernel_vec[i + j * KW + c*KH*KW]);
            }
          printf(" / ");
        }
        printf("\n");
      }
#endif        

  }

  int32_t *inp_c[C];
  for (int c = 0; c < C; c++)
  {
    inp_c[c] = mm_out_c + c*out_w*out_h;
  }
  
  #pragma omp parallel for num_threads(4)
  for (int i = 0; i < out_h; i++)
  {
    for (int j = 0; j < out_w; j++)
    {      
        *inp_c[0] = *inp_c[0]+*inp_c[1]+*inp_c[2];
        inp_c[0]++;
        inp_c[1]++;
        inp_c[2]++;
    }
  }

  #pragma omp parallel for num_threads(4)
  for(int c=0;c<C;c++) {
    for (dim_t d = 0; d < KN; d++) { 
       
      vector_scale_bias_offset_clip_s8(mm_out_c + d *out_w*out_h, *((int32_t*)bias+d), output  + (d*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+1)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+1)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+2)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+2)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
      //vector_scale_bias_offset_clip_s8(mm_out_c + (d+3)*out_w*out_h, *((int32_t*)bias+d), output  + ((d+3)*out_h*out_w) , scale, bscale, out_h * out_w , true, doRelu,outOffset);
  
    }
  }
    

  clock_t t2e = clock();
  cout <<"nn + sc: : "<<(double)(t2e- t2s)/CLOCKS_PER_SEC*1000 << std::endl;
    
      
      
#if DEBUG_INFO
    if(d<KN) 
    {
        
        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < 2; i++)
            {
                printf("%d ",mm_out_temp[i + j * out_w]+ *((int32_t*)bias+d));

            }
            printf("\n");
        }
        
    }
#endif



#if DEBUG_INFO
    if(d<KN) 
    {
        
        for (int j = 0; j < 2; j++)
        {
            for (int i = 0; i < 2; i++)
            {
                printf("(%d)",*(output  + (d*out_h*out_w) + i + j * out_w) );
                if(i % 16 == 15) printf(" / ");
            }
            printf("\n");
        }
        
    }
#endif    


  
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


  //TODO: scale converting?
  float filterScale = kernelScale;
  filterScale = 1/filterScale;
  float inScale = inputScale;
  inScale = 1/inScale;
  float biScale = biasScale;
  biScale = 1/biScale;
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

#if DEBUG_INFO
  printf("tempscale:%f scale:%f matmulscale:%f\n", tempScale, scale, matMulScale);
#endif  

  if (C % 16 == 0 && KN % 16 == 0 && tempScale == scale && biScale == matMulScale && outputOffset == 0 && inputOffset == 0 && kernelOffset == 0 && biasOffset == 0) {

        printf("++ fwdVTAConv ++: NHWC:%d,%d,%d,%d pad:%d, stride:%d scale i:%f o:%f f:%f, C:%d KH:%d KW:%d KN:%d relu=%d\n",N,H,W,C,pad_size, stride_size, inputScale,outputScale,kernelScale,C,KH,KW,KN,doRelu);

        llvm::SmallVector<dim_t, 4> inputDims = {N, H, W, C};
        Type inputType(ElemKind::Int8QTy, inputDims, inputScale, inputOffset);
        Tensor inV((void *)input, &inputType);

        llvm::SmallVector<dim_t, 4> outputDims = {N, (dim_t)out_h, (dim_t)out_w, KN};
  Type outputType(ElemKind::Int8QTy, outputDims, outputScale, outputOffset);
        Tensor outV((void *)output, &outputType);

        llvm::SmallVector<dim_t, 4> kernelDims = {KN, KH, KW, C};
  Type kernelType(ElemKind::Int8QTy, kernelDims, kernelScale, kernelOffset);
        Tensor filterV((void *)kernel, &kernelType);

        Tensor *biasV;
        if (doBias) {
    llvm::SmallVector<dim_t, 1> biasDims = {KN};
    Type biasType(ElemKind::Int32QTy, biasDims, biasScale, biasOffset);
          biasV = new Tensor((void *)bias, &biasType);
        } else {
    biasV = new Tensor(ElemKind::Int32QTy, {KN}, 1.0, 0);
    biasV->zero();
  }

        std::array<unsigned_t, 2> kernelS = {(unsigned_t)KH, (unsigned_t)KW};
  llvm::ArrayRef<unsigned_t> kernelSizes(kernelS);

        std::array<unsigned_t, 2> strideS = {(unsigned_t)stride_size,
                                            (unsigned_t)stride_size};
  llvm::ArrayRef<unsigned_t> strides(strideS);

        std::array<unsigned_t, 4> padS = {(unsigned_t)pad_size, (unsigned_t)pad_size,
                                          (unsigned_t)pad_size, (unsigned_t)pad_size};

  llvm::ArrayRef<unsigned_t> pads(padS);

        fwdVTAConvolutionInstQuantizedImpl(&inV, &outV, &filterV, biasV,
                                       kernelSizes, strides, pads, group,
                                       dilation, doRelu);
  }else {

     printf("-- fwdVTAConv --: NHWC:%d,%d,%d,%d pad:%d, stride:%d scale i:%f o:%f b:%f f:%f, C:%d KH:%d KW:%d KN:%d \n",N,H,W,C,pad_size, stride_size, inputScale,outputScale,biasScale, kernelScale,C,KH,KW,KN);

    // convert to nchw
    int8_t* input_nchw = (int8_t*)malloc(N*H*W*C);
    int8_t* k_nchw = (int8_t*)malloc(C*KH*KW*KN);
    int8_t* output_nchw = (int8_t*)malloc(N*out_w*out_h*KN);

      if(C==3 && W%16==0) {
       nhwc_to_nchw_s8_c3_w16n(input,input_nchw,N,H,W,C);   
       printf("channel 3 => nhwc_to_nchw_s8_c3\n");
      } else {
        nhwc_to_nchw_s8(input,input_nchw,N,H,W,C);   
        printf("nhwc_to_nchw_s8\n");
      } 

      //TODO change to neon
      // kernel nhwc -> nchw
      for(int kn=0;kn<KN;kn++)
      {
        for(int j=0;j<KH;j++)
        {
          for(int i=0;i<KW;i++)
          {
            for(int c=0;c<C;c++)
            {
              k_nchw[kn*KH*KW*C + c*KW*KH + j*KW + i ] = kernel[ kn*KH*KW*C + j*KW*C + i*C + c];
            }
          }
        }
      }
#if DEBUG_INFO


      printf("NCHW_conv\n");
      for(int j=0;j<H*C;j+=16)
      {
        for(int i=0;i<20;i++) printf("%02x ", input_nchw[i+j*W]) ;
        printf("\n%d) ",j+16);
      }

#endif      
/* temporary disabled. mm version not complete
    if(KN%4==0) { // && out_w*out_hKH*KW*C%16==0
      printf("MM PATH");
      fwdConvolution_nchw_mm(input_nchw, k_nchw, bias,output_nchw, 
                inputScale, kernelScale, biasScale, outputScale, inputOffset, kernelOffset, biasOffset, outputOffset,
                                       N,C,H,W, KN, KH, KW,  pad_size, stride_size, 1,
                                       dilation, doRelu, doBias, out_h, out_w);

    }
    else {
*/
      printf("VEC PATH");
    fwdConvolution_nchw(input_nchw, k_nchw, bias,output_nchw, 
                inputScale, kernelScale, biasScale, outputScale, inputOffset, kernelOffset, biasOffset, outputOffset,
                                       N,C,H,W, KN, KH, KW,  pad_size, stride_size, 1,
                                       dilation, doRelu, doBias, out_h, out_w);

//    }


    //TODO change to neon
    // output nchw -> nhwc && row-major <-> column-major
    for(int n=0;n<N;n++)
    {
      for(int j=0;j<out_h;j++)
      {
        for(int i=0;i<out_w;i++)
  {
          for(int kn=0;kn<KN;kn++)
    {

            output[n*out_w*out_h*KN + i*out_w*KN + j*KN + kn] = output_nchw[ n*out_w*out_h*KN + kn*out_w*out_h + j*out_w + i];
          }
        }
    }
  }

  free(input_nchw);
  free(k_nchw);
  free(output_nchw);

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
      for(int h=0,ho=0;ho < out_h;h+=stride_size,ho++)
      {
        for(int w=0,wo=0;wo< out_w ;w+=stride_size,wo++)
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
  int out_w = ((W+2*p + 1) - (KW - 1))/stride_size;
  int out_h = ((H+2*p +1) - (KH - 1))/stride_size;

  const float scale = 1.0f * (inputScale / outputScale / (KW*KH)); 

#if DEBUG_INFO
  printf("scale:%f\n",scale);
#endif  

  for(int n=0;n<N;n++)
  {
    for(int c=0;c<C/16;c++) 
    {
        for(int h=0,ho=0;ho<out_h;h+=stride_size,ho++) // ho = h out.
        {
          for(int w=0,wo=0;wo<out_w;w+=stride_size,wo++)
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