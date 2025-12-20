#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class ShemetovDIncreaseContrastFunctionalTests : public ::testing::Test {
 protected:
  InType input_data;
  OutType expected_output;
  int width = 0;
  int height = 0;
  int channels = 0;

  void SetUp() override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    MPI_Bcast(input_data.data(), static_cast<int>(total), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    expected_output.resize(total);
    constexpr float kFactor = 1.3F;
    for (size_t i = 0; i < total; ++i) {
      const int v = static_cast<int>(static_cast<float>(input_data[i]) * kFactor);
      expected_output[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  }
};

TEST_F(ShemetovDIncreaseContrastFunctionalTests, SeqFullCycle) {
  IncreaseContrastTaskSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), expected_output);
}

TEST_F(ShemetovDIncreaseContrastFunctionalTests, MpiFullCycle) {
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

TEST_F(ShemetovDIncreaseContrastFunctionalTests, SeqRunOnly) {
  IncreaseContrastTaskSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
}

TEST_F(ShemetovDIncreaseContrastFunctionalTests, MpiRunOnly) {
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

TEST(ShemetovDIncreaseContrastConstructorTests, SeqConstructorInitializesCorrectly) {
  InType input = {1, 2, 3, 4, 5};
  IncreaseContrastTaskSEQ task(input);

  EXPECT_EQ(task.GetInput().size(), static_cast<size_t>(5));
  EXPECT_EQ(task.GetOutput().size(), static_cast<size_t>(5));
  EXPECT_TRUE(task.Validation());
}

TEST(ShemetovDIncreaseContrastConstructorTests, MpiConstructorInitializesCorrectly) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = {10, 20, 30, 40, 50};
  IncreaseContrastTaskMPI task(input);

  if (rank == 0) {
    EXPECT_EQ(task.GetInput().size(), static_cast<size_t>(5));
    EXPECT_EQ(task.GetOutput().size(), static_cast<size_t>(5));
    EXPECT_TRUE(task.Validation());
  }
}

TEST(ShemetovDIncreaseContrastProcessingTests, SeqPreProcessingReturnsTrue) {
  InType input = {100, 150, 200};
  IncreaseContrastTaskSEQ task(input);

  ASSERT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(ShemetovDIncreaseContrastProcessingTests, SeqPostProcessingReturnsTrue) {
  InType input = {100, 150, 200};
  IncreaseContrastTaskSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(ShemetovDIncreaseContrastProcessingTests, MpiPreProcessingReturnsTrue) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = {100, 150, 200};
  IncreaseContrastTaskMPI task(input);

  ASSERT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
}

TEST(ShemetovDIncreaseContrastProcessingTests, MpiPostProcessingReturnsTrue) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType input = {100, 150, 200};
  IncreaseContrastTaskMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(ShemetovDIncreaseContrastEdgeCases, SeqZeroAndMax) {
  InType zero_input(10, 0);
  InType max_input(10, 255);

  IncreaseContrastTaskSEQ task_zero(zero_input);
  if (task_zero.Validation()) {
    ASSERT_TRUE(task_zero.PreProcessing());
    ASSERT_TRUE(task_zero.Run());
    EXPECT_TRUE(
        std::all_of(task_zero.GetOutput().begin(), task_zero.GetOutput().end(), [](uint8_t v) { return v == 0; }));
  } else {
    SUCCEED();
  }

  IncreaseContrastTaskSEQ task_max(max_input);
  if (task_max.Validation()) {
    ASSERT_TRUE(task_max.PreProcessing());
    ASSERT_TRUE(task_max.Run());
    EXPECT_TRUE(
        std::all_of(task_max.GetOutput().begin(), task_max.GetOutput().end(), [](uint8_t v) { return v == 255; }));
  } else {
    SUCCEED();
  }
}

TEST(ShemetovDIncreaseContrastEdgeCases, MpiZeroAndMax) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType zero_input(10, 0);
  InType max_input(10, 255);

  IncreaseContrastTaskMPI task_zero(zero_input);
  ASSERT_TRUE(task_zero.Validation());
  ASSERT_TRUE(task_zero.PreProcessing());
  ASSERT_TRUE(task_zero.Run());
  ASSERT_TRUE(task_zero.PostProcessing());
  if (rank == 0) {
    EXPECT_TRUE(
        std::all_of(task_zero.GetOutput().begin(), task_zero.GetOutput().end(), [](uint8_t v) { return v == 0; }));
  }

  IncreaseContrastTaskMPI task_max(max_input);

  ASSERT_TRUE(task_max.Validation());
  ASSERT_TRUE(task_max.PreProcessing());
  ASSERT_TRUE(task_max.Run());
  ASSERT_TRUE(task_max.PostProcessing());

  if (rank == 0) {
    EXPECT_TRUE(
        std::all_of(task_max.GetOutput().begin(), task_max.GetOutput().end(), [](uint8_t v) { return v == 255; }));
  }
}

TEST(ShemetovDIncreaseContrastEdgeCases, SeqEmptyInput) {
  InType empty;
  IncreaseContrastTaskSEQ task(empty);

  EXPECT_FALSE(task.Validation());
  EXPECT_EQ(task.GetInput().size(), static_cast<size_t>(0));
  EXPECT_EQ(task.GetOutput().size(), static_cast<size_t>(0));
}

TEST(ShemetovDIncreaseContrastEdgeCases, MpiEmptyInput) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType empty;
  IncreaseContrastTaskMPI task(empty);

  EXPECT_FALSE(task.Validation());

  if (rank == 0) {
    EXPECT_EQ(task.GetInput().size(), static_cast<size_t>(0));
    EXPECT_EQ(task.GetOutput().size(), static_cast<size_t>(0));
  }
}

TEST(ShemetovDIncreaseContrastEdgeCases, SeqSingleElement) {
  InType data = {128};
  IncreaseContrastTaskSEQ task(data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput()[0], static_cast<uint8_t>(std::clamp(int(128 * 1.3F), 0, 255)));
}

TEST(ShemetovDIncreaseContrastEdgeCases, MpiSingleElement) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType data = {200};
  IncreaseContrastTaskMPI task(data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    EXPECT_EQ(task.GetOutput()[0], static_cast<uint8_t>(std::clamp(int(200 * 1.3F), 0, 255)));
  } else {
    SUCCEED();
  }
}

TEST(ShemetovDIncreaseContrastEdgeCases, MpiUnevenSizes) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  InType data(static_cast<size_t>((size * 2) + 1), 100);
  IncreaseContrastTaskMPI task(data);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (rank == 0) {
    for (uint8_t v : task.GetOutput()) {
      EXPECT_EQ(v, static_cast<uint8_t>(std::clamp(int(100 * 1.3F), 0, 255)));
    }
  } else {
    SUCCEED();
  }
}

TEST(ShemetovDIncreaseContrastEdgeCases, SeqClampBehavior) {
  InType low_values = {0, 1, 50, 100, 150, 200, 254, 255};
  IncreaseContrastTaskSEQ task(low_values);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());

  for (uint8_t v : task.GetOutput()) {
    EXPECT_GE(v, 0);
    EXPECT_LE(v, 255);
  }

  EXPECT_EQ(task.GetOutput()[0], 0);
  EXPECT_EQ(task.GetOutput()[7], 255);
}

TEST(ShemetovDIncreaseContrastEdgeCases, MpiClampBehavior) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  InType values = {0, 100, 200, 255};
  IncreaseContrastTaskMPI task(values);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());

  if (rank == 0) {
    for (uint8_t v : task.GetOutput()) {
      EXPECT_GE(v, 0);
      EXPECT_LE(v, 255);
    }
  }
}

}  // namespace shemetov_d_increasing_contrast
