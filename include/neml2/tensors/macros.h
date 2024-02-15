// Copyright 2023, UChicago Argonne, LLC
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

#define typedef_all_FixedDimTensor_prefix(classname, prefix)                                       \
  typedef classname<Scalar> prefix##Scalar;                                                        \
  typedef classname<Vec> prefix##Vec;                                                              \
  typedef classname<Rot> prefix##Rot;                                                              \
  typedef classname<R2> prefix##R2;                                                                \
  typedef classname<SR2> prefix##SR2;                                                              \
  typedef classname<R3> prefix##R3;                                                                \
  typedef classname<SFR3> prefix##SFR3;                                                            \
  typedef classname<R4> prefix##R4;                                                                \
  typedef classname<SSR4> prefix##SSR4;                                                            \
  typedef classname<SFFR4> prefix##SFFR4;                                                          \
  typedef classname<R5> prefix##R5;                                                                \
  typedef classname<SSFR5> prefix##SSFR5;                                                          \
  typedef classname<WR2> prefix##WR2

#define typedef_all_FixedDimTensor_suffix(classname, suffix)                                       \
  typedef classname<Scalar> Scalar##suffix;                                                        \
  typedef classname<Vec> Vec##suffix;                                                              \
  typedef classname<Rot> Rot##suffix;                                                              \
  typedef classname<R2> R2##suffix;                                                                \
  typedef classname<SR2> SR2##suffix;                                                              \
  typedef classname<R3> R3##suffix;                                                                \
  typedef classname<SFR3> SFR3##suffix;                                                            \
  typedef classname<R4> R4##suffix;                                                                \
  typedef classname<SSR4> SSR4##suffix;                                                            \
  typedef classname<SFFR4> SFFR4##suffix;                                                          \
  typedef classname<R5> R5##suffix;                                                                \
  typedef classname<SSFR5> SSFR5##suffix;                                                          \
  typedef classname<WR2> WR2##suffix

#define instantiate_all_FixedDimTensor(classname)                                                  \
  template class classname<Scalar>;                                                                \
  template class classname<Vec>;                                                                   \
  template class classname<Rot>;                                                                   \
  template class classname<R2>;                                                                    \
  template class classname<SR2>;                                                                   \
  template class classname<R3>;                                                                    \
  template class classname<SFR3>;                                                                  \
  template class classname<R4>;                                                                    \
  template class classname<SSR4>;                                                                  \
  template class classname<SFFR4>;                                                                 \
  template class classname<R5>;                                                                    \
  template class classname<SSFR5>;                                                                 \
  template class classname<WR2>

#define register_all_FixedDimTensor_prefix(prefix1, prefix2)                                       \
  register_NEML2_object_alias(prefix1##Scalar, prefix2 "Scalar");                                  \
  register_NEML2_object_alias(prefix1##Vec, prefix2 "Vec");                                        \
  register_NEML2_object_alias(prefix1##Rot, prefix2 "Rot");                                        \
  register_NEML2_object_alias(prefix1##R2, prefix2 "R2");                                          \
  register_NEML2_object_alias(prefix1##SR2, prefix2 "SR2");                                        \
  register_NEML2_object_alias(prefix1##R3, prefix2 "R3");                                          \
  register_NEML2_object_alias(prefix1##SFR3, prefix2 "SFR3");                                      \
  register_NEML2_object_alias(prefix1##R4, prefix2 "R4");                                          \
  register_NEML2_object_alias(prefix1##SSR4, prefix2 "SSR4");                                      \
  register_NEML2_object_alias(prefix1##SFFR4, prefix2 "SFFR4");                                    \
  register_NEML2_object_alias(prefix1##R5, prefix2 "R5");                                          \
  register_NEML2_object_alias(prefix1##SSFR5, prefix2 "SSFR5");                                    \
  register_NEML2_object_alias(prefix1##WR2, prefix2 "WR2")

#define register_all_FixedDimTensor_suffix(suffix1, suffix2)                                       \
  register_NEML2_object_alias(Scalar##suffix1, "Scalar" suffix2);                                  \
  register_NEML2_object_alias(Vec##suffix1, "Vec" suffix2);                                        \
  register_NEML2_object_alias(Rot##suffix1, "Rot" suffix2);                                        \
  register_NEML2_object_alias(R2##suffix1, "R2" suffix2);                                          \
  register_NEML2_object_alias(SR2##suffix1, "SR2" suffix2);                                        \
  register_NEML2_object_alias(R3##suffix1, "R3" suffix2);                                          \
  register_NEML2_object_alias(SFR3##suffix1, "SFR3" suffix2);                                      \
  register_NEML2_object_alias(R4##suffix1, "R4" suffix2);                                          \
  register_NEML2_object_alias(SSR4##suffix1, "SSR4" suffix2);                                      \
  register_NEML2_object_alias(SFFR4##suffix1, "SFFR4" suffix2);                                    \
  register_NEML2_object_alias(R5##suffix1, "R5" suffix2);                                          \
  register_NEML2_object_alias(SSFR5##suffix1, "SSFR5" suffix2);                                    \
  register_NEML2_object_alias(WR2##suffix1, "WR2" suffix2)

#define FOR_ALL_FIXEDDIMTENSOR(f)                                                                  \
  f(Scalar);                                                                                       \
  f(Vec);                                                                                          \
  f(Rot);                                                                                          \
  f(R2);                                                                                           \
  f(SR2);                                                                                          \
  f(R3);                                                                                           \
  f(SFR3);                                                                                         \
  f(R4);                                                                                           \
  f(SSR4);                                                                                         \
  f(SFFR4);                                                                                        \
  f(R5);                                                                                           \
  f(SSFR5);                                                                                        \
  f(WR2)
