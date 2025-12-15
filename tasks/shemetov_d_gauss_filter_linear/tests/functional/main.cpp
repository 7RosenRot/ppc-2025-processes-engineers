#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"
#include "shemetov_d_gauss_filter_linear/mpi/include/ops_mpi.hpp"
#include "shemetov_d_gauss_filter_linear/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace shemetov_d_gauss_filter_linear {

class GaussFilterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string name = std::get<0>(test_param);
    for (auto &c : name) {
      if (std::isalnum(static_cast<unsigned char>(c)) == 0) {
        c = '_';
      }
    }
    return name;
  }

 protected:
  void SetUp() override {
    const std::string img_path = "tasks/shemetov_d_gauss_filter_linear/data/pic.jpg";

    int width = -1;
    int height = -1;
    int channels = -1;
    uint8_t *data = stbi_load(img_path.c_str(), &width, &height, &channels, STBI_rgb);
    if (data == nullptr) {
      throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
    }
    channels = STBI_rgb;
    std::vector<uint8_t> raw_data(data, data + (static_cast<ptrdiff_t>(width) * height * channels));
    stbi_image_free(data);

    input_data_.resize(height, std::vector<uint8_t>(width));
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        input_data_[i][j] = raw_data[(i * width) + j];
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && output_data.size() == input_data_.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(GaussFilterFuncTests, ApplyFilter) {
  ExecuteTest(GetParam());
}

TEST(GaussFilterExtraFuncTests, SmallSyntheticImageSEQ) {
  InType input(5, std::vector<uint8_t>(5, 10));
  input[2][2] = 200;

  GaussFilterSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &out = task.GetOutput();

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_LE(out[i][j], 255);
    }
  }
}

TEST(GaussFilterExtraFuncTests, SmallSyntheticImageMPI) {
  InType input(5, std::vector<uint8_t>(5, 10));
  input[2][2] = 200;

  GaussFilterMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const auto &out = task.GetOutput();

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_LE(out[i][j], 255);
    }
  }
}

const std::array<TestType, 1> kTestParam = {std::make_tuple("pic.jpg")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_shemetov_d_gauss_filter_linear),
    ppc::util::AddFuncTask<GaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_shemetov_d_gauss_filter_linear));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaussFilterFuncTests::PrintFuncTestName<GaussFilterFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicGaussFilterTests, GaussFilterFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace shemetov_d_gauss_filter_linear
