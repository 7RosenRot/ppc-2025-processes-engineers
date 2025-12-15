#include "shemetov_d_gauss_filter_linear/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "shemetov_d_gauss_filter_linear/common/include/common.hpp"

namespace shemetov_d_gauss_filter_linear {

GaussFilterMPI::GaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

float GaussFilterMPI::ApplyKernel(const InType &in, int i, int j, const std::vector<std::vector<float>> &kernel) {
  float sum = 0.F;
  for (int ki = -1; ki <= 1; ++ki) {
    for (int kj = -1; kj <= 1; ++kj) {
      sum += kernel[ki + 1][kj + 1] * static_cast<float>(in[i + ki][j + kj]);
    }
  }
  return sum;
}

bool GaussFilterMPI::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty();
}

bool GaussFilterMPI::PreProcessingImpl() {
  return true;
}

bool GaussFilterMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  auto &out = GetOutput();
  const int height = static_cast<int>(in.size());
  const int width = static_cast<int>(in[0].size());

  const std::vector<std::vector<float>> kernel = {
      {1.F / 16, 2.F / 16, 1.F / 16}, {2.F / 16, 4.F / 16, 2.F / 16}, {1.F / 16, 2.F / 16, 1.F / 16}};

  const int block_size = height / size;
  const int start_row = rank * block_size;
  const int end_row = (rank == size - 1) ? height : start_row + block_size;

  const auto img_size = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  std::vector<uint8_t> local_out(img_size, 0);

  for (int i = std::max(1, start_row); i < std::min(end_row - 1, height - 1); ++i) {
    for (int j = 1; j < width - 1; ++j) {
      local_out[(i * width) + j] = static_cast<uint8_t>(std::clamp(ApplyKernel(in, i, j, kernel), 0.F, 255.F));
    }
  }

  std::vector<uint8_t> global_out(img_size, 0);

  MPI_Allreduce(local_out.data(), global_out.data(), size, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      out[i][j] = global_out[(i * width) + j];
    }
  }

  return true;
}

bool GaussFilterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_gauss_filter_linear
