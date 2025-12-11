#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "shemetov_d_increasing_contrast/common/include/common.hpp"

namespace shemetov_d_increasing_contrast {

IncreaseContrastTaskMPI::IncreaseContrastTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool IncreaseContrastTaskMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool IncreaseContrastTaskMPI::PreProcessingImpl() {
  return true;
}

bool IncreaseContrastTaskMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const size_t total_size = GetInput().size();

  std::vector<int> counts(size, static_cast<int>(total_size / size));
  const int rest = static_cast<int>(total_size % size);
  for (int i = 0; i < rest; ++i) {
    counts[i]++;
  }

  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  std::vector<uint8_t> local_in(counts[rank]);
  std::vector<uint8_t> local_out(counts[rank]);

  MPI_Scatterv(GetInput().data(), counts.data(), displs.data(), MPI_UNSIGNED_CHAR, local_in.data(), counts[rank],
               MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  constexpr float kFactor = 1.3F;

  for (size_t i = 0; i < local_in.size(); ++i) {
    const int v = static_cast<int>(local_in[i] * kFactor);
    local_out[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
  }

  MPI_Gatherv(local_out.data(), counts[rank], MPI_UNSIGNED_CHAR, GetOutput().data(), counts.data(), displs.data(),
              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool IncreaseContrastTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_increasing_contrast
