#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class ShemetovDIncreaseContrastPerformanceTests : public ::testing::Test {
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

TEST_F(ShemetovDIncreaseContrastPerformanceTests, SeqFullCycle) {
  IncreaseContrastTaskSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), expected_output);
}

TEST_F(ShemetovDIncreaseContrastPerformanceTests, MpiFullCycle) {
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

TEST_F(ShemetovDIncreaseContrastPerformanceTests, SeqRunOnly) {
  IncreaseContrastTaskSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());

  EXPECT_EQ(task.GetOutput(), expected_output);
}

TEST_F(ShemetovDIncreaseContrastPerformanceTests, MpiRunOnly) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  IncreaseContrastTaskMPI task(input_data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());

  if (rank == 0) {
    EXPECT_EQ(task.GetOutput(), expected_output);
  } else {
    SUCCEED();
  }
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, SeqSmallData) {
  InType data(1000, 50);
  IncreaseContrastTaskSEQ task(data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput()[0], static_cast<uint8_t>(std::clamp(int(50 * 1.3F), 0, 255)));
  EXPECT_EQ(task.GetOutput().size(), static_cast<size_t>(1000));
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, MpiSmallData) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType data(1000, 150);

  IncreaseContrastTaskMPI task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_EQ(task.GetOutput()[0], static_cast<uint8_t>(std::clamp(int(150 * 1.3F), 0, 255)));
    EXPECT_EQ(task.GetOutput().size(), static_cast<size_t>(1000));
  } else {
    SUCCEED();
  }
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, SeqLargeData) {
  constexpr size_t kLargeSize = 10'000'000;
  InType data(kLargeSize, 200);

  IncreaseContrastTaskSEQ task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());

  EXPECT_EQ(task.GetOutput().size(), kLargeSize);
  EXPECT_EQ(task.GetOutput()[0], static_cast<uint8_t>(std::clamp(int(200 * 1.3F), 0, 255)));
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, MpiLargeData) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  constexpr size_t kLargeSize = 10'000'000;
  InType data(kLargeSize, 200);

  IncreaseContrastTaskMPI task(data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());

  if (rank == 0) {
    EXPECT_EQ(task.GetOutput().size(), kLargeSize);
    EXPECT_EQ(task.GetOutput()[0], static_cast<uint8_t>(std::clamp(int(200 * 1.3F), 0, 255)));
  } else {
    SUCCEED();
  }
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, SeqVariousSizes) {
  std::vector<size_t> sizes = {100, 1000, 10000, 100000, 1000000};

  for (size_t size : sizes) {
    InType data(size, 128);
    IncreaseContrastTaskSEQ task(data);

    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());

    EXPECT_EQ(task.GetOutput().size(), size);
  }
}

TEST(ShemetovDIncreaseContrastPerformanceAdditionalTests, MpiVariousSizes) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<size_t> sizes = {100, 1000, 10000, 100000, 1000000};

  for (size_t size : sizes) {
    InType data(size, 128);
    IncreaseContrastTaskMPI task(data);

    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());

    if (rank == 0) {
      EXPECT_EQ(task.GetOutput().size(), size);
    }
  }
}

}  // namespace shemetov_d_increasing_contrast
