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

#include "neml2/tensors/user_tensors/Orientation.h"

#include "neml2/tensors/Quaternion.h"

using namespace torch::indexing;

namespace neml2
{
register_NEML2_object(Orientation);

OptionSet
Orientation::expected_options()
{
  OptionSet options = UserTensor::expected_options();
  options.set<std::string>("input_type") = "euler_angles";
  options.set<std::string>("angle_convention") = "kocks";
  options.set<std::string>("angle_type") = "degrees";
  options.set<std::vector<Real>>("values") = {};
  options.set<bool>("normalize") = false;

  options.set<TorchSize>("random_seed") = -1;

  options.set<unsigned int>("quantity") = 1;
  return options;
}

Orientation::Orientation(const OptionSet & options)
  : Rot(fill(options)),
    UserTensor(options)
{
}

Rot
Orientation::fill(const OptionSet & options) const
{
  std::string input_type = options.get<std::string>("input_type");

  Rot R;
  if (input_type == "euler_angles")
  {
    R = expand_as_needed(fill_euler_angles(torch::tensor(options.get<std::vector<Real>>("values"),
                                                         default_tensor_options()),
                                           options.get<std::string>("angle_convention"),
                                           options.get<std::string>("angle_type")),
                         options.get<unsigned int>("quantity"));
  }
  else if (input_type == "random")
  {
    R = fill_random(options.get<unsigned int>("quantity"), options.get<TorchSize>("random_seed"));
  }
  else
    throw NEMLException("Unknown Orientation input_type " + input_type);

  if (options.get<bool>("normalize"))
    return math::where((R.norm_sq() < 1.0).unsqueeze(-1), R, R.shadow());

  return R;
}

Rot
Orientation::fill_euler_angles(const torch::Tensor & vals,
                               std::string angle_convention,
                               std::string angle_type) const
{
  neml_assert(
      (torch::numel(vals) % 3) == 0,
      "Orientation input values should have length divisable by 3 for input type 'euler_angles'");
  auto ten = vals.reshape({-1, 3});

  if (angle_type == "degrees")
    ten = torch::deg2rad(ten);
  else
    neml_assert(angle_type == "radians",
                "Orientation angle_type must be either 'degrees' or 'radians'");

  if (angle_convention == "bunge")
  {
    ten.index_put_({Ellipsis, 0}, torch::fmod(ten.index({Ellipsis, 0}) - M_PI / 2.0, 2.0 * M_PI));
    ten.index_put_({Ellipsis, 1}, torch::fmod(ten.index({Ellipsis, 1}), M_PI));
    ten.index_put_({Ellipsis, 2}, torch::fmod(M_PI / 2.0 - ten.index({Ellipsis, 2}), 2.0 * M_PI));
  }
  else if (angle_convention == "roe")
  {
    ten.index_put_({Ellipsis, 2}, M_PI - ten.index({Ellipsis, 2}));
  }
  else
    neml_assert(angle_convention == "kocks",
                "Unknown Orientation angle_convention " + angle_convention);

  // Make a rotation matrix...
  auto M = torch::zeros({ten.sizes()[0], 3, 3}, vals.options());
  auto a = ten.index({Ellipsis, 0});
  auto b = ten.index({Ellipsis, 1});
  auto c = ten.index({Ellipsis, 2});

  M.index_put_({Ellipsis, 0, 0},
               -torch::sin(c) * torch::sin(a) - torch::cos(c) * torch::cos(a) * torch::cos(b));
  M.index_put_({Ellipsis, 0, 1},
               torch::sin(c) * torch::cos(a) - torch::cos(c) * torch::sin(a) * torch::cos(b));
  M.index_put_({Ellipsis, 0, 2}, torch::cos(c) * torch::sin(b));
  M.index_put_({Ellipsis, 1, 0},
               torch::cos(c) * torch::sin(a) - torch::sin(c) * torch::cos(a) * torch::cos(b));
  M.index_put_({Ellipsis, 1, 1},
               -torch::cos(c) * torch::cos(a) - torch::sin(c) * torch::sin(a) * torch::cos(b));
  M.index_put_({Ellipsis, 1, 2}, torch::sin(c) * torch::sin(b));
  M.index_put_({Ellipsis, 2, 0}, torch::cos(a) * torch::sin(b));
  M.index_put_({Ellipsis, 2, 1}, torch::sin(a) * torch::sin(b));
  M.index_put_({Ellipsis, 2, 2}, torch::cos(b));

  // Convert from matrix to vector
  return fill_matrix(R2(M, 1));
}

Rot
Orientation::fill_matrix(const R2 & M) const
{
  // Get the angle
  auto trace = M.index({Ellipsis, 0, 0}) + M.index({Ellipsis, 1, 1}) + M.index({Ellipsis, 2, 2});
  auto theta = torch::acos((trace - 1.0) / 2.0);

  // Get the standard Rod. parameters
  auto scale = torch::tan(theta / 2.0) / (2.0 * torch::sin(theta));
  scale.index_put_({theta == 0}, 0.0);
  auto rx = (M.index({Ellipsis, 2, 1}) - M.index({Ellipsis, 1, 2})) * scale;
  auto ry = (M.index({Ellipsis, 0, 2}) - M.index({Ellipsis, 2, 0})) * scale;
  auto rz = (M.index({Ellipsis, 1, 0}) - M.index({Ellipsis, 0, 1})) * scale;

  return fill_rodrigues(rx, ry, rz);
}

Rot
Orientation::fill_rodrigues(const Scalar & rx, const Scalar & ry, const Scalar & rz) const
{
  // Get the modified Rod. parameters
  auto ns = rx * rx + ry * ry + rz * rz;
  auto f = torch::sqrt(torch::Tensor(ns) + torch::tensor(1.0, ns.dtype())) +
           torch::tensor(1.0, ns.dtype());

  // Stack and return
  return Rot(torch::stack({rx / f, ry / f, rz / f}, 1), 1);
}

Rot
Orientation::fill_random(unsigned int n, TorchSize random_seed) const
{
  if (random_seed >= 0)
    torch::manual_seed(random_seed);
  auto u0 = torch::rand({n}, default_tensor_options());
  auto u1 = torch::rand({n}, default_tensor_options());
  auto u2 = torch::rand({n}, default_tensor_options());

  auto w = torch::sqrt(1.0 - u0) * torch::sin(2.0 * M_PI * u1);
  auto x = torch::sqrt(1.0 - u0) * torch::cos(2.0 * M_PI * u1);
  auto y = torch::sqrt(u0) * torch::sin(2.0 * M_PI * u2);
  auto z = torch::sqrt(u0) * torch::cos(2.0 * M_PI * u2);

  auto quats = Quaternion(torch::stack({w, x, y, z}, 1), 1);

  return fill_matrix(quats.to_R2());
}

Rot
Orientation::expand_as_needed(const Rot & input, unsigned int inp_size) const
{
  if (inp_size > 1)
    return input.batch_expand({inp_size});
  else
    return input;
}

} // namespace neml2
