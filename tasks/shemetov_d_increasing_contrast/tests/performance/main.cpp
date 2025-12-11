#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastPerformanceTests : public ::testing::Test {
 protected:
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    const size_t N = 1'000'000;
    input_data_.assign(N, 128);

    expected_output_.resize(N);
    const float factor = 1.3f;
    for (size_t i = 0; i < N; ++i) {
      int v = static_cast<int>(128 * factor);
      expected_output_[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  }
};

TEST_F(IncreaseContrastPerformanceTests, SEQ_Perf) {
  IncreaseContrastTaskSEQ task(input_data_);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), expected_output_);
}

TEST_F(IncreaseContrastPerformanceTests, MPI_Perf) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  IncreaseContrastTaskMPI task(input_data_);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_EQ(task.GetOutput(), expected_output_);
  } else {
    SUCCEED();
  }
}

}  // namespace shemetov_d_increasing_contrast
