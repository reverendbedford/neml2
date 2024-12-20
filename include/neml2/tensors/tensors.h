// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/SFR3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SFR4.h"
#include "neml2/tensors/WFR4.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/SFFR4.h"
#include "neml2/tensors/R5.h"
#include "neml2/tensors/SSFR5.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/Quaternion.h"
#include "neml2/tensors/SWR4.h"
#include "neml2/tensors/WSR4.h"
#include "neml2/tensors/WWR4.h"
#include "neml2/models/crystallography/MillerIndex.h"
#include "neml2/tensors/SSSSR8.h"
#include "neml2/tensors/R8.h"

#include "neml2/tensors/macros.h"

#include <iostream>

namespace neml2
{
#define _tensor_type_enum(T) k##T

// Enum for tensor type introspection
enum class TensorType : int8_t
{
  FOR_ALL_TENSORBASE_COMMA(_tensor_type_enum),
  kUknown
};

template <typename T>
struct TensorTypeEnum
{
  static constexpr TensorType value = TensorType::kUknown;
};

// Specialize TensorEnum for all tensor types
#define _tensor_type_enum_specialize(T)                                                            \
  template <>                                                                                      \
  struct TensorTypeEnum<T>                                                                         \
  {                                                                                                \
    static constexpr TensorType value = TensorType::k##T;                                          \
  }
FOR_ALL_TENSORBASE(_tensor_type_enum_specialize);

// Stringify the tensor type enum
std::ostream & operator<<(std::ostream & os, const TensorType & t);
}
