#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <cstdint>
#include <cstddef>    // size_t

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastFunctionalTests : public ::testing::Test {
 protected:
  InType inputData;
  OutType expectedOutput;

  void SetUp() override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int w = 0;
    int h = 0;
    int ch = 0;

    unsigned char* data = nullptr;
    const std::string imgPath = "tasks/shemetov_d_increasing_contrast/data/pic.jpg";

    if (rank == 0) {
      data = stbi_load(imgPath.c_str(), &w, &h, &ch, STBI_rgb);
      ASSERT_TRUE(data != nullptr) << "Failed to load image: " << imgPath;
    }

    MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ch, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t total =
        static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(ch);

    inputData.resize(total);

    MPI_Bcast(
        reinterpret_cast<unsigned char*>(inputData.data()),
        static_cast<int>(total),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD);

    if (rank == 0) {
      stbi_image_free(data);
    }

    expectedOutput.resize(total);

    constexpr float factor = 1.3F;
    for (size_t i = 0; i < total; ++i) {
      const int v = static_cast<int>(inputData[i] * factor);
      expectedOutput[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  }
};

TEST_F(IncreaseContrastFunctionalTests, SeqTest) {
  IncreaseContrastTaskSEQ task(inputData);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), expectedOutput);
}

TEST_F(IncreaseContrastFunctionalTests, MpiTest) {
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
