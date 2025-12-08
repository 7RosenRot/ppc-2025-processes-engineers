#include <gtest/gtest.h>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"
#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shemetov_d_increasing_contrast {

class IncreaseContrastPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_.resize(kCount_, 128);
    expected_output_.resize(kCount_);
    std::transform(input_data_.begin(), input_data_.end(), expected_output_.begin(),
                   [](uint8_t pixel) { return static_cast<uint8_t>(std::clamp(int(pixel * 1.3f), 0, 255)); });
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(IncreaseContrastPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, IncreaseContrastTaskMPI, IncreaseContrastTaskSEQ>(
    PPC_SETTINGS_shemetov_d_increasing_contrast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = IncreaseContrastPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ContrastPerfTests, IncreaseContrastPerfTests, kGtestValues, kPerfTestName);

}  // namespace shemetov_d_increasing_contrast
