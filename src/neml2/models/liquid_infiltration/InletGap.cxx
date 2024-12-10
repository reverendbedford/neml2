#include "neml2/models/liquid_infiltration/InletGap.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(InletGap);
OptionSet
InletGap::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<CrossRef<Scalar>>("Initial_Porosity");
  options.set<CrossRef<Scalar>>("Product_Thickness_Growth_Ratio");
  options.set<VariableName>("Product_Thickness") = VariableName("state", "delta");
  options.set<VariableName>("Inlet_Gap") = VariableName("state", "r1");
  return options;
}

InletGap::InletGap(const OptionSet & options)
  : Model(options),
    _M(declare_parameter<Scalar>("M",
                                 "Product_Thickness_Growth_Ratio")), //, /*allow_nonlinear=*/true)),
    _phi0(declare_parameter<Scalar>("phi0", "Initial_Porosity")),    //,/*allow_nonlinear=*/true)),
    _sqrtd(declare_input_variable<Scalar>("Product_Thickness")),
    _r1(declare_output_variable<Scalar>("Inlet_Gap"))
{
}

void
InletGap::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _r1 = math::sqrt(_phi0) - _M * _sqrtd * _sqrtd;
  }

  if (dout_din)
  {
    // const auto * const M = nl_param("M");
    // const auto * const phi0 = nl_param("phi0");
    // if (M)
    //     _r1.d(*M) = -(_sqrtd*_sqrtd);

    // if (phi0)
    //     _r1.d(*phi0) = 1.0/(2.0*math::sqrt(_phi0));
    _r1.d(_sqrtd) = -2.0 * _M * _sqrtd;
  }
}
}