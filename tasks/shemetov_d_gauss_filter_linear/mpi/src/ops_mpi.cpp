#include "shemetov_d_gauss_filter_linear/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace shemetov_d_gauss_filter_linear {

GaussFilterMPI::GaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool GaussFilterMPI::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty();
}

bool GaussFilterMPI::PreProcessingImpl() {
  return true;
}

bool GaussFilterMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &in = GetInput();
  auto &out = GetOutput();
  int height = static_cast<int>(in.size());
  int width = static_cast<int>(in[0].size());

  const std::vector<std::vector<float>> kernel = {
      {1.f / 16, 2.f / 16, 1.f / 16}, {2.f / 16, 4.f / 16, 2.f / 16}, {1.f / 16, 2.f / 16, 1.f / 16}};

  int block_size = height / size;
  int start_row = rank * block_size;
  int end_row = (rank == size - 1) ? height : start_row + block_size;

  for (int i = std::max(1, start_row); i < std::min(end_row - 1, height - 1); i++) {
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

  MPI_Allreduce(MPI_IN_PLACE, out.data()->data(), height * width, MPI_UINT8_T, MPI_MAX, MPI_COMM_WORLD);

  return true;
}

bool GaussFilterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_gauss_filter_linear
