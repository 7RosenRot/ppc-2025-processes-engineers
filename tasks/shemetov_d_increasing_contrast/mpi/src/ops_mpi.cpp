#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>

namespace shemetov_d_increasing_contrast {

IncreaseContrastTaskMPI::IncreaseContrastTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size(), 0);
}

bool IncreaseContrastTaskMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool IncreaseContrastTaskMPI::PreProcessingImpl() {
  return true;
}

bool IncreaseContrastTaskMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t total_pixels = GetInput().size();
  size_t chunk = total_pixels / size;
  size_t start = rank * chunk;
  size_t end = (rank == size - 1) ? total_pixels : start + chunk;

  const float factor = 1.3f;
  for (size_t i = start; i < end; ++i) {
    int tmp = static_cast<int>(GetInput()[i] * factor);
    GetOutput()[i] = static_cast<uint8_t>(std::clamp(tmp, 0, 255));
  }

  // Gather данные с всех процессов на процесс 0
  if (rank != 0) {
    MPI_Send(GetOutput().data() + start, end - start, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD);
  } else {
    for (int r = 1; r < size; ++r) {
      size_t r_start = r * chunk;
      size_t r_end = (r == size - 1) ? total_pixels : r_start + chunk;
      MPI_Recv(GetOutput().data() + r_start, r_end - r_start, MPI_UINT8_T, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return !GetOutput().empty();
}

bool IncreaseContrastTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_increasing_contrast
