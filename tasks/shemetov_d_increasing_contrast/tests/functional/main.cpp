#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastFunctionalTests : public ::testing::Test {
 protected:
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    int w = 0, h = 0, ch = 0;

    std::string img_path = "tasks/shemetov_d_increasing_contrast/data/pic.jpg";

    unsigned char *data = stbi_load(img_path.c_str(), &w, &h, &ch, STBI_rgb);
    ASSERT_TRUE(data != nullptr) << "Failed to load image: " << img_path;

    size_t total = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    input_data_.assign(data, data + static_cast<ptrdiff_t>(total));
    stbi_image_free(data);

    expected_output_.resize(total);
    const float factor = 1.3f;
    for (size_t i = 0; i < total; ++i) {
      int v = static_cast<int>(input_data_[i] * factor);
      expected_output_[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  }
};

TEST_F(IncreaseContrastFunctionalTests, SEQ_Test) {
  IncreaseContrastTaskSEQ task(input_data_);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), expected_output_);
}

TEST_F(IncreaseContrastFunctionalTests, MPI_Test) {
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
