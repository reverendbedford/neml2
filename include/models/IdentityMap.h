#pragma once

#include "models/Model.h"

// input -> output identity map
template <typename T>
class IdentityMap : public Model
{
public:
  IdentityMap(const std::string & name, const std::string & axis_name, const std::string & var_name)
    : Model(name),
      _axis_name(axis_name),
      _var_name(var_name)
  {
    input().add<LabeledAxis>(_axis_name);
    input().subaxis(_axis_name).add<T>(_var_name);

    output().add<LabeledAxis>(_axis_name);
    output().subaxis(_axis_name).add<T>(_var_name);

    setup();
  }

  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const
  {
    out.slice(0, _axis_name).set(in.slice(0, _axis_name)(_var_name), _var_name);
    if (dout_din)
    {
      auto I = T::identity_map().expand_batch(in.batch_size());
      dout_din->block(_axis_name, _axis_name).set(I, _var_name, _var_name);
    }
  }

protected:
  std::string _axis_name;
  std::string _var_name;
};
