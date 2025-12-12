#include "shemetov_d_gauss_filter_linear/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace shemetov_d_gauss_filter_linear {

GaussFilterSEQ::GaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
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
      {1.f / 16, 2.f / 16, 1.f / 16}, {2.f / 16, 4.f / 16, 2.f / 16}, {1.f / 16, 2.f / 16, 1.f / 16}};

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      float sum = 0.f;
      for (int ki = -1; ki <= 1; ki++) {
        for (int kj = -1; kj <= 1; kj++) {
          sum += kernel[ki + 1][kj + 1] * in[i + ki][j + kj];
        }
      }
      out[i][j] = static_cast<uint8_t>(std::clamp(sum, 0.f, 255.f));
    }
  }

  return true;
}

bool GaussFilterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_gauss_filter_linear
