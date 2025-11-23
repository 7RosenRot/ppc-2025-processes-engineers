#include "shemetov_d_find_error_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>

namespace shemetov_d_find_error_vec {

namespace {

constexpr double kEpsilon = 1e-10;

}  // namespace

int ShemetovDFindErrorVecMPI::DetectDrop(double left, double right) noexcept {
  return (left > right + kEpsilon) ? 1 : 0;
}

ShemetovDFindErrorVecMPI::ShemetovDFindErrorVecMPI(const InType& input) {
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
  const auto& data = GetInput();
  const int dataSize = static_cast<int>(data.size());

  if (dataSize < 2) {
    GetOutput() = 0;
    return true;
  }

  int worldRank = 0, worldSize = 1;

  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  if (dataSize <= worldSize) {
    int localResult = 0;

    if (worldRank == 0) {
      for (int i = 0; i < dataSize - 1; i += 1) {
        localResult += DetectDrop(data[i], data[i + 1]);
      }
    }

    MPI_Bcast(&localResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = localResult;

    return true;
  }

  const int baseChunk = dataSize / worldSize;
  const int extraChunk = dataSize % worldSize;

  const int begin = worldRank * baseChunk + std::min(worldRank, extraChunk);

  const int count = baseChunk + (worldRank < extraChunk ? 1 : 0);

  const int end = begin + count;

  assert(end <= dataSize);

  int localViolations = 0;

  for (int i = begin; i + 1 < end; i += 1) {
    localViolations += DetectDrop(data[i], data[i + 1]);
  }

  if (worldRank > 0 && begin > 0) {
    localViolations += DetectDrop(data[begin - 1], data[begin]);
  }

  int globalViolations = 0;

  MPI_Allreduce(&localViolations, &globalViolations, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = globalViolations;
  return true;
}

bool ShemetovDFindErrorVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shemetov_d_find_error_vec
