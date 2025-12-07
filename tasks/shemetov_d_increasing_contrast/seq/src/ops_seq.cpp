#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>

namespace shemetov_d_increasing_contrast {

IncreaseContrastTaskSEQ::IncreaseContrastTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size(), 0);
}

bool IncreaseContrastTaskSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool IncreaseContrastTaskSEQ::PreProcessingImpl() {
  // Для этой задачи предварительная обработка не требуется
  return true;
}

bool IncreaseContrastTaskSEQ::RunImpl() {
  const float factor = 1.3f;

  std::transform(GetInput().begin(), GetInput().end(), GetOutput().begin(), [factor](uint8_t pixel) {
    int tmp = static_cast<int>(pixel * factor);
    return static_cast<uint8_t>(std::clamp(tmp, 0, 255));
  });
  return !GetOutput().empty();
}

bool IncreaseContrastTaskSEQ::PostProcessingImpl() {
  // Постобработка не требуется
  return true;
}

}  // namespace shemetov_d_increasing_contrast
