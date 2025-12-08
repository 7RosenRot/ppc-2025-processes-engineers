#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shemetov_d_increasing_contrast {

using InType = std::vector<uint8_t>;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shemetov_d_increasing_contrast
