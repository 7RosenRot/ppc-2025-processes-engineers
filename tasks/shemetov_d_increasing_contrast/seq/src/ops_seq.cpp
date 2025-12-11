#include "shemetov_d_increasing_contrast/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>

namespace shemetov_d_increasing_contrast {

IncreaseContrastTaskSEQ::IncreaseContrastTaskSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool IncreaseContrastTaskSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool IncreaseContrastTaskSEQ::PreProcessingImpl() {
  return true;
}

bool IncreaseContrastTaskSEQ::RunImpl() {
  constexpr float factor = 1.3F;

  for (size_t i = 0; i < GetInput().size(); ++i) {
    const int value = static_cast<int>(GetInput()[i] * factor);
    GetOutput()[i] = static_cast<uint8_t>(std::clamp(value, 0, 255));
  }

  return true;
}

bool IncreaseContrastTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_increasing_contrast
