#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>  // for size_t
#include <cstdint>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastPerformanceTests : public ::testing::Test {
 protected:
  InType inputData;
  OutType expectedOutput;

  void SetUp() override {
    constexpr size_t n = 1'000'000;
    inputData.assign(n, 128);

    expectedOutput.resize(n);
    constexpr float factor = 1.3F;

    for (size_t i = 0; i < n; ++i) {
      const int v = static_cast<int>(128 * factor);
      expectedOutput[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  }
};

TEST_F(IncreaseContrastPerformanceTests, SeqPerf) {
  IncreaseContrastTaskSEQ task(inputData);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), expectedOutput);
}

TEST_F(IncreaseContrastPerformanceTests, MpiPerf) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  IncreaseContrastTaskMPI task(inputData);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_EQ(task.GetOutput(), expectedOutput);
  } else {
    SUCCEED();
  }
}

}  // namespace shemetov_d_increasing_contrast
