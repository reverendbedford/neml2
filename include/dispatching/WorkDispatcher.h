#pragma once

#include "tensors/BatchTensor.h"

/// Information about the work progress, i.e. how many batches of a batched tensor have been
/// completed.
struct Progress
{
  Progress(TorchSize t)
    : current(0),
      total(t)
  {
  }

  TorchSize current;
  TorchSize total;
};

/// The lambda f has the signature
///   void f(TorchSize begin, TorchSize end);
/// The lambda shall slice the batched tensor with the range and perform the operations.
template <typename F>
inline void
chunked_map(F && f, TorchSize chunk_size, Progress & progress)
{
  TorchSize current, next;
  while (progress.current < progress.total)
  {
    {
      // Some scoped spin mutex lock here
      current = progress.current;
      next = current + chunk_size;
      if (next > progress.total)
        next = progress.total;
      progress.current = next;
    }
    f(current, next);
  }
}

/// The lambda f has the signature
///   void f(TorchSize begin, TorchSize end);
/// The lambda shall slice the batched tensor with the range, transfer those batches to GPU, perform
/// the operations, and finally transfer the data back to CPU.
template <typename F>
inline void
cuda_chunked_map(F && f, TorchSize chunk_size, Progress & progress)
{
  TorchSize current, next;
  // This thread does the following:

  // 1. Do an MPI communication to figure which CPU has fallen behind (e.g. has the smallest
  //    progress.current/progress.total).

  // TODO...

  // 2. If that lucky CPU is me, then do the following.

  {
    // Some scoped spin mutex lock here
    current = progress.current;
    next = current + chunk_size;
    if (next > progress.total)
      next = progress.total;
    progress.current = next;
  }
  f(current, next);
}
