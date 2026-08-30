// Standalone probe: does CUTLASS at this checkout provide a *simple rowwise*
// (non-block-scaled) FP8 GEMM CollectiveBuilder specialization for
// cutlass::arch::Sm103 (GB300), the same style fp8_gemm.cu's existing
// DeviceGemmFp8RowwiseSm100 struct uses for Sm100 (B200)?
//
// Compile only (no torch/pip involved) -- fastest possible signal:
//   nvcc -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
//     -I<cutlass>/include -I<cutlass>/tools/util/include \
//     -gencode=arch=compute_103a,code=sm_103a \
//     -c test_sm103_rowwise.cu -o /tmp/test_sm103_rowwise.o
//
// If this compiles: CUTLASS has a usable non-block-scaled Sm103 path and we
// can write DeviceGemmFp8RowwiseSm103 in fp8_gemm.cu the same way Sm100 is
// written. If it fails to compile (not just fails at runtime), CUTLASS's
// CollectiveBuilder has no rowwise specialization for Sm103 at this commit
// -- only the sm103_blockscaled_* kernels exist, and rowwise FP8 genuinely
// isn't available for GB300 without a different quantization scheme.

#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.hpp>
#include <cute/tensor.hpp>

using namespace cute;

using ElementType = cutlass::float_e4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;
using OutElementType = cutlass::bfloat16_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;
static constexpr int AlignmentD = AlignmentC;

using TileShape = Shape<_256, _128, _64>;
using ClusterShape = Shape<_2, _2, _1>;

using MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm103,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    void,
    LayoutC,
    AlignmentC,
    OutElementType,
    LayoutD,
    AlignmentD,
    EpilogueScheduleType>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm103,
    cutlass::arch::OpClassTensorOp,
    ElementType,
    LayoutA,
    AlignmentA,
    ElementType,
    LayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopScheduleType>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int main() {
  printf("SM103 rowwise CollectiveBuilder compiled OK. sizeof(GemmKernel)=%zu\n", sizeof(GemmKernel));
  return 0;
}
