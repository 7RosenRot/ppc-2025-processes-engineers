#include "shemetov_d_gauss_filter_linear/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"

namespace shemetov_d_gauss_filter_linear {

GaussFilterSEQ::GaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

float GaussFilterSEQ::ApplyKernel(const InType &in, int i, int j, const std::vector<std::vector<float>> &kernel) {
  float sum = 0.F;
  for (int ki = -1; ki <= 1; ++ki) {
    for (int kj = -1; kj <= 1; ++kj) {
      sum += kernel[ki + 1][kj + 1] * static_cast<float>(in[i + ki][j + kj]);
    }
  }
  return sum;
}

bool GaussFilterSEQ::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty();
}

bool GaussFilterSEQ::PreProcessingImpl() {
  return true;
}

bool GaussFilterSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  int height = static_cast<int>(in.size());
  int width = static_cast<int>(in[0].size());

  const std::vector<std::vector<float>> kernel = {
      {1.F / 16, 2.F / 16, 1.F / 16}, {2.F / 16, 4.F / 16, 2.F / 16}, {1.F / 16, 2.F / 16, 1.F / 16}};

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      float val = ApplyKernel(in, i, j, kernel);
      if (val < 0.F || val > 255.F) {
        return false;
      }
      out[i][j] = static_cast<uint8_t>(std::clamp(val, 0.F, 255.F));
    }
  }

  return true;
}

bool GaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_gauss_filter_linear
