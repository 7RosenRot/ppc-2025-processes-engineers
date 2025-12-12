#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastPerformanceTests : public ::testing::Test {
 protected:
  InType input_data;
  OutType expected_output;

  void SetUp() override {
    constexpr size_t kNumElements = 1'000'000;
    input_data.assign(kNumElements, 128);

    expected_output.resize(kNumElements);
    constexpr float kFactor = 1.3F;
    for (size_t i = 0; i < kNumElements; ++i) {
      const int v = static_cast<int>(128 * kFactor);
      expected_output[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  }
};

TEST_F(IncreaseContrastPerformanceTests, SeqRun) {
  IncreaseContrastTaskSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), expected_output);
}

TEST_F(IncreaseContrastPerformanceTests, MpiRun) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  IncreaseContrastTaskMPI task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_EQ(task.GetOutput(), expected_output);
  } else {
    SUCCEED();
  }
}

TEST(IncreaseContrastPerformanceAdditionalTests, SeqSmall) {
  InType data(1000, 50);
  IncreaseContrastTaskSEQ task(data);
  task.Validation();
  task.Run();
  EXPECT_EQ(task.GetOutput()[0], static_cast<uint8_t>(std::clamp(int(50 * 1.3f), 0, 255)));
}

TEST(IncreaseContrastPerformanceAdditionalTests, MpiSmall) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType data(1000, 150);
  IncreaseContrastTaskMPI task(data);
  task.Validation();
  task.Run();

  if (rank == 0) {
    EXPECT_EQ(task.GetOutput()[0], static_cast<uint8_t>(std::clamp(int(150 * 1.3f), 0, 255)));
  } else {
    SUCCEED();
  }
}

}  // namespace shemetov_d_increasing_contrast
