#include "shemetov_d_find_error_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <vector>

namespace shemetov_d_find_error_vec {

namespace {

constexpr double epsilon = 1e-10;

}  // namespace

int ShemetovDFindErrorVecMPI::DetectDrop(double left, double right) noexcept {
  return (left > right + epsilon) ? 1 : 0;
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
  if (data_size <= world_size) {
    int local_result = 0;

    if (world_rank == 0) {
      for (int i = 0; i < data_size - 1; i += 1) {
        local_result += DetectDrop(data[i], data[i + 1]);
      }
    }

    MPI_Bcast(&local_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = local_result;

    return true;
  }

  const int base_chunk = data_size / world_size;
  const int extra_chunk = data_size % world_size;

  const int begin = (world_rank * base_chunk) + std::min(world_rank, extra_chunk);

  const int count = base_chunk + (world_rank < extra_chunk ? 1 : 0);

  const int end = begin + count;

  assert(end <= data_size);

  int local_violations = 0;

  for (int i = begin; i + 1 < end; i += 1) {
    local_violations += DetectDrop(data[i], data[i + 1]);
  }

  if (world_rank > 0 && begin > 0) {
    local_violations += DetectDrop(data[begin - 1], data[begin]);
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
