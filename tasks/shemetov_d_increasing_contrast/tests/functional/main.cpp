#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastFunctionalTests : public ::testing::Test {
 protected:
  InType input_data;
  OutType expected_output;

  void SetUp() override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char *data = nullptr;

    const std::string img_path = "tasks/shemetov_d_increasing_contrast/data/pic.jpg";

    if (rank == 0) {
      data = stbi_load(img_path.c_str(), &width, &height, &channels, STBI_rgb);
      ASSERT_TRUE(data != nullptr) << "Failed to load image: " << img_path;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height) * channels;

    input_data.resize(total);

    if (rank == 0) {
      std::copy(data, data + total, input_data.begin());
      stbi_image_free(data);
    }

    MPI_Bcast(input_data.data(), static_cast<int>(total), MPI_UINT8_T, 0, MPI_COMM_WORLD);

    expected_output.resize(total);
    constexpr float kFactor = 1.3F;
    for (size_t i = 0; i < total; ++i) {
      const int v = static_cast<int>(input_data[i] * kFactor);
      expected_output[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  }
};

TEST_F(IncreaseContrastFunctionalTests, SeqRun) {
  IncreaseContrastTaskSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), expected_output);
}

TEST_F(IncreaseContrastFunctionalTests, MpiRun) {
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

TEST(IncreaseContrastEdgeCases, SeqZeroAndMax) {
  InType zero_input(10, 0);
  InType max_input(10, 255);

  IncreaseContrastTaskSEQ task_zero(zero_input);
  ASSERT_TRUE(task_zero.Validation());
  task_zero.Run();
  EXPECT_TRUE(
      std::all_of(task_zero.GetOutput().begin(), task_zero.GetOutput().end(), [](uint8_t v) { return v == 0; }));

  IncreaseContrastTaskSEQ task_max(max_input);
  ASSERT_TRUE(task_max.Validation());
  task_max.Run();
  EXPECT_TRUE(
      std::all_of(task_max.GetOutput().begin(), task_max.GetOutput().end(), [](uint8_t v) { return v == 255; }));
}

TEST(IncreaseContrastEdgeCases, MpiZeroAndMax) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType zero_input(10, 0);
  InType max_input(10, 255);

  IncreaseContrastTaskMPI task_zero(zero_input);
  task_zero.Validation();
  task_zero.Run();
  if (rank == 0) {
    EXPECT_TRUE(
        std::all_of(task_zero.GetOutput().begin(), task_zero.GetOutput().end(), [](uint8_t v) { return v == 0; }));
  }

  IncreaseContrastTaskMPI task_max(max_input);
  task_max.Validation();
  task_max.Run();
  if (rank == 0) {
    EXPECT_TRUE(
        std::all_of(task_max.GetOutput().begin(), task_max.GetOutput().end(), [](uint8_t v) { return v == 255; }));
  }
}

}  // namespace shemetov_d_increasing_contrast
