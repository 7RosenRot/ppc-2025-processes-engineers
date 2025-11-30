#include "shemetov_d_find_error_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "shemetov_d_find_error_vec/common/include/common.hpp"

namespace shemetov_d_find_error_vec {

namespace {

constexpr double kEpsilon = 1e-10;

}  // namespace

int ShemetovDFindErrorVecMPI::DetectDrop(double left, double right) noexcept {
  return (left > right + kEpsilon) ? 1 : 0;
}

ShemetovDFindErrorVecMPI::ShemetovDFindErrorVecMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0;
}

bool ShemetovDFindErrorVecMPI::ValidationImpl() {
  return true;
}

bool ShemetovDFindErrorVecMPI::PreProcessingImpl() {
  return true;
}

bool ShemetovDFindErrorVecMPI::RunImpl() {
  const auto &data = GetInput();
  const int data_size = static_cast<int>(data.size());

  if (data_size < 2) {
    GetOutput() = 0;
    return true;
  }

  int world_rank = 0;
  int world_size = 1;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<int> sendcounts(world_size, 0);
  std::vector<int> displs(world_size, 0);

  const int base = data_size / world_size;
  const int extra = data_size % world_size;

  for (int r = 0; r < world_size; ++r) {
    sendcounts[r] = base + (r < extra ? 1 : 0);
    displs[r] = (r * base) + std::min(r, extra);
  }

  const int local_size = sendcounts[world_rank];
  std::vector<double> local_data(local_size);

  MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_data.data(), local_size, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  int local_violations = 0;

  for (int i = 0; i + 1 < local_size; i++) {
    local_violations += DetectDrop(local_data[i], local_data[i + 1]);
  }

  if (world_rank > 0 && displs[world_rank] > 0) {
    const double left = data[displs[world_rank] - 1];
    const double right = local_data[0];
    local_violations += DetectDrop(left, right);
  }

  int global_violations = 0;

  MPI_Allreduce(&local_violations, &global_violations, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_violations;
  return true;
}

bool ShemetovDFindErrorVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_find_error_vec
