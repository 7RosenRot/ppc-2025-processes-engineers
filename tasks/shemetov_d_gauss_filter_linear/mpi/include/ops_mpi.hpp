#pragma once
#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shemetov_d_gauss_filter_linear {

class GaussFilterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit GaussFilterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shemetov_d_gauss_filter_linear
