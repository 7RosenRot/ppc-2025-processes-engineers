#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"
#include "shemetov_d_gauss_filter_linear/mpi/include/ops_mpi.hpp"
#include "shemetov_d_gauss_filter_linear/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shemetov_d_gauss_filter_linear {

class GaussFilterPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int size = 512;
    input_data_.resize(size, std::vector<uint8_t>(size, 128));
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

TEST_P(GaussFilterPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

TEST(GaussFilterPerfExtraTest, SmallMatrixSEQ) {
  InType input(8, std::vector<uint8_t>(8, 100));
  input[4][4] = 255;

  GaussFilterSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(GaussFilterPerfExtraTest, SmallMatrixMPI) {
  const int size = 64;
  InType input(size, std::vector<uint8_t>(size, 100));
  input[size / 2][size / 2] = 255;

  GaussFilterMPI task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaussFilterMPI, GaussFilterSEQ>(PPC_SETTINGS_shemetov_d_gauss_filter_linear);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaussFilterPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GaussFilterPerfTest, kGtestValues, kPerfTestName);

}  // namespace shemetov_d_gauss_filter_linear
