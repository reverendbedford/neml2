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

#include "neml2/models/Model.h"

namespace neml2
{
/**
 * @brief This model is a blender that mixes together a matrix of models by interpolating the input
 * domain.
 *
 * The matrix of models must share the same input and output axis. (This constraint could be relaxed
 * a little bit...)
 *
 * The blended model is "tabulated" using two user selected variables, e.g., temperature and von
 * mises stress. That is, consider the following grid,
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *                        \  Temperature (K):   (-infty, 600)     [600, 1000)    [1000, infty)
 *  von Mises stress (MPa) \
 *                          \_________________________________________________________________
 *                          |
 *                          |
 *    (-infty, 20)          |                      Model 11          Model 12        Model 13
 *                          |
 *                          |
 *                          |
 *    [20, 80)              |                      Model 21          Model 22        Model 23
 *                          |
 *                          |
 *                          |
 *    [80, infty)           |                      Model 31          Model 32        Model 33
 *                          |
 *                          |
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Model 11, 12, 13, 21, 22, 23, 31, 32, 33 are the 9 models in the matrix. This class "lumps" all
 * models together with a smooth 2 dimensional interpolating function. The interpolating function is
 * essentially $\delta_{ij}$ where i and j are matrix indices. The smoothing is done similar to the
 * sigmoid function.
 */
class MatrixModel : public Model
{
public:
  MatrixModel(const OptionSet & options);

  static OptionSet expected_options();

protected:
  void set_value(bool, bool, bool) override;

  std::vector<std::vector<Model *>> _matrix;
};
}
