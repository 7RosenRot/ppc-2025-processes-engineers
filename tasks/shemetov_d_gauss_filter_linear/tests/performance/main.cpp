#include <gtest/gtest.h>

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

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaussFilterMPI, GaussFilterSEQ>(PPC_SETTINGS_shemetov_d_gauss_filter_linear);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GaussFilterPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GaussFilterPerfTest, kGtestValues, kPerfTestName);

}  // namespace shemetov_d_gauss_filter_linear
