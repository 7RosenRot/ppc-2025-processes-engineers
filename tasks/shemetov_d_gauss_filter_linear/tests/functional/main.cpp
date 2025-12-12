#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"
#include "shemetov_d_gauss_filter_linear/mpi/include/ops_mpi.hpp"
#include "shemetov_d_gauss_filter_linear/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shemetov_d_gauss_filter_linear {

class GaussFilterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string name = std::get<0>(test_param);
    for (auto &c : name) {
      if (!isalnum(c)) {
        c = '_';
      }
    }
    return name;
  }

 protected:
  void SetUp() override {
    const std::string img_path = "tasks/shemetov_d_increasing_contrast/data/pic.jpg";

    int width = -1, height = -1, channels = -1;
    uint8_t *data = stbi_load(img_path.c_str(), &width, &height, &channels, STBI_rgb);
    if (!data) {
      throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
    }
    channels = STBI_rgb;
    std::vector<uint8_t> raw_data(data, data + (width * height * channels));
    stbi_image_free(data);

    input_data_.resize(height, std::vector<uint8_t>(width));
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        input_data_[i][j] = raw_data[i * width + j];
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

const std::array<TestType, 1> kTestParam = {std::make_tuple("pic.jpg")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<GaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_shemetov_d_gauss_filter_linear),
    ppc::util::AddFuncTask<GaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_shemetov_d_gauss_filter_linear));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GaussFilterFuncTests::PrintFuncTestName<GaussFilterFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicGaussFilterTests, GaussFilterFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace shemetov_d_gauss_filter_linear
