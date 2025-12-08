#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
  public:
  static std::string PrintTestParam(const TestType& value) {
    // TestType = std::tuple<std::string>
    const std::string& name = std::get<0>(value);

    // Убираем запрещённые символы
    std::string sanitized = name;
    std::replace_if(sanitized.begin(), sanitized.end(),
                    [](char c){ return !std::isalnum(c); }, '_');

    return sanitized;
  }
 
  protected:
  void SetUp() override {
    int width = -1, height = -1, channels = -1;
    std::vector<uint8_t> img;

    std::string abs_path = "data/pic.jpg";
    auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_rgb);
    if (!data) {
      throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
    }
    channels = STBI_rgb;
    img = std::vector<uint8_t>(data, data + static_cast<ptrdiff_t>(width * height * channels));
    stbi_image_free(data);

    input_data_ = img;

    expected_output_.resize(input_data_.size());
    std::transform(input_data_.begin(), input_data_.end(), expected_output_.begin(),
                   [](uint8_t pixel) { return static_cast<uint8_t>(std::clamp(int(pixel * 1.3f), 0, 255)); });
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(IncreaseContrastFuncTests, ApplyContrast) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::make_tuple("pic.jpg")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<IncreaseContrastTaskMPI, InType>(kTestParam, PPC_SETTINGS_shemetov_d_increasing_contrast),
    ppc::util::AddFuncTask<IncreaseContrastTaskSEQ, InType>(kTestParam, PPC_SETTINGS_shemetov_d_increasing_contrast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = IncreaseContrastFuncTests::PrintFuncTestName<IncreaseContrastFuncTests>;

INSTANTIATE_TEST_SUITE_P(ContrastTests, IncreaseContrastFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace shemetov_d_increasing_contrast
