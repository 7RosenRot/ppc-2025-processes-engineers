#include "shemetov_d_increasing_contrast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t total = GetInput().size();

  // --- Подсчёт количества элементов ---
  std::vector<int> counts(size, total / size);
  int rest = total % size;
  for (int i = 0; i < rest; i++) {
    counts[i]++;
  }

  // --- Смещения ---
  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; i++) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  // --- Локальные буферы ---
  std::vector<uint8_t> local_in(counts[rank]);
  std::vector<uint8_t> local_out(counts[rank]);

  // --- Рассылка данных ---
  MPI_Scatterv(GetInput().data(), counts.data(), displs.data(), MPI_UINT8_T, local_in.data(), counts[rank], MPI_UINT8_T,
               0, MPI_COMM_WORLD);

  // --- Простая обработка ---
  const float factor = 1.3f;
  for (size_t i = 0; i < local_in.size(); i++) {
    int v = static_cast<int>(local_in[i] * factor);
    local_out[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
  }

  // --- Сбор результата ---
  MPI_Gatherv(local_out.data(), counts[rank], MPI_UINT8_T, GetOutput().data(), counts.data(), displs.data(),
              MPI_UINT8_T, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool IncreaseContrastTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_increasing_contrast
