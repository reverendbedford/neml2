# Model definition {#modeldefinition}

In NEML2, a model is a function
\f[
  f: \mathbb{R}^m \to \mathbb{R}^n
\f]
mapping from the input space \f$\mathbb{R}^m\f$ of dimension \f$m\f$ to the output space \f$\mathbb{R}^n\f$ of dimension \f$n\f$. The input vector is the concatenation of \f$p\f$ flattened variables, i.e.,
\f[
  x = \left[ x_i \right]_{i=1}^p \in \mathbb{R}^m, \quad \sum_{i=1}^p \lvert x_i \rvert = m.
\f]
Similarly, the output vector is the concatenation of \f$q\f$ flattened variables, i.e.,
\f[
  y = \left[ y_i \right]_{i=1}^q \in \mathbb{R}^n, \quad \sum_{i=1}^q \lvert y_i \rvert = n.
\f]

Translating the above mathematical definition into NEML2 is straightforward.
- A model following this definition derives from [Model](@ref neml2::Model).
- [declare_input_variable](@ref neml2::Model::declare_input_variable) declares an input variable \f$x_i\f$ in the input space \f$\mathbb{R}^m\f$.
- [declare_output_variable](@ref neml2::Model::declare_output_variable) declares an output variable \f$y_i\f$ in the output space \f$\mathbb{R}^n\f$.
- [set_value](@ref neml2::Model::set_value) is a method you must override to define the forward operator \f$f\f$.

Both `declare_input_variable` and `declare_output_variable` are templated on the variable type -- recall that only a variable of the NEML2 primitive tensor type can be registered. Furthermore, both calls return a convenient accessor of type [LabeledAxisAccessor](@ref neml2::LabeledAxisAccessor) which can be later used to retrieve/modify the labeled view of the input/output vector.

> Declaration of the variables don't immediately set up the layout of the input/output `LabeledAxis`. The method [setup](@ref neml2::Model::setup) should be explicitly called in order to set up the memory layout of the `LabeledAxis`s. **Note that [setup](@ref neml2::Model::setup) must be called after all the variables have been added, and before the forward operator of the `Model` can be used.**
