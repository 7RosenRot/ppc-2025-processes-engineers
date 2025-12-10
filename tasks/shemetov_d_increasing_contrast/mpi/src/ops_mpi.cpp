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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const float factor = 1.3f;

  const auto &input = GetInput();
  size_t total = input.size();

  size_t chunk = total / size;
  size_t remainder = total % size;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  for (int r = 0; r < size; ++r) {
    sendcounts[r] = chunk + (r == size - 1 ? remainder : 0);
    displs[r] = r * chunk;
  }

  size_t local_n = sendcounts[rank];
  std::vector<uint8_t> local_in(local_n);

  MPI_Scatterv(input.data(), sendcounts.data(), displs.data(), MPI_UINT8_T, local_in.data(), local_n, MPI_UINT8_T, 0,
               MPI_COMM_WORLD);

  std::vector<uint8_t> local_out(local_n);
  for (size_t i = 0; i < local_n; ++i) {
    int tmp = int(local_in[i] * factor);
    local_out[i] = uint8_t(std::clamp(tmp, 0, 255));
  }

  if (rank == 0) {
    GetOutput().assign(total, 0);
  } else {
    GetOutput().clear();
  }

  MPI_Gatherv(local_out.data(), local_n, MPI_UINT8_T, rank == 0 ? GetOutput().data() : nullptr, sendcounts.data(),
              displs.data(), MPI_UINT8_T, 0, MPI_COMM_WORLD);

  return true;
}

bool IncreaseContrastTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_increasing_contrast
