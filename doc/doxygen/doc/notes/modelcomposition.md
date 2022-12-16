# Model composition {#modelcomposition}

[TOC]

Quoting [Wikipedia](https://en.wikipedia.org/wiki/Function_composition):
> In mathematics, function composition is an operation \f$\circ\f$ that takes two functions \f$f\f$ and \f$g\f$, and produces a function \f$h = g \circ f\f$ such that \f$h(x) = g(f(x))\f$.

Since NEML2 `Model` is a function (in the context of mathematics) at its core, it should be possible, in theory, to compose different NEML2 `Model`s into a new NEML2 `Model`. Yes, this is possible -- the [ComposedModel](@ref neml2::ComposedModel) is for that precise purpose.

Similar to the statement "a composed function is a function" in the context of mathematics, in NEML2, the equivalent statement "a `ComposedModel` is a `Model`" also holds. In addition, the `ComposedModel` provides four key features to help simplify the composition and reduces computational cost:
- Automatic dependency registration
- Automatic input/output identification
- Automatic dependency resolution
- Automatic chain rule

## A symbolic example

To demonstrate the utility of the four key features of `ComposedModel`, let us consider the composition of three functions \f$f\f$, \f$g\f$, and \f$h\f$:
\f{align*}
  y_1 &= f(x_1, x_2), \\
  y_2 &= g(y_1), \\
  y &= h(y_1, y_2).
\f}

## Automatic dependency registration

It is obvious to us that the function \f$h\f$ _depends_ on functions \f$f\f$ and \f$g\f$ because the input of \f$h\f$ depends on the outputs of \f$f\f$ and \f$g\f$. Such dependency is automatically identified and registered while composing a `ComposedModel` in NEML2. This procedure is called "automatic dependency registration".

In order to identify dependencies among different `Model`s, we keep track of the set of _consumed_ variables, \f$\mathcal{I}_i\f$, and a set of _provided_ variables, \f$\mathcal{O}_i\f$, for each `Model` \f$f_i\f$. When a set of models (functions) are composed together, `Model` \f$f_i\f$ is said to _depend_ on \f$f_j\f$ if and only if \f$\exists x\f$ such that
\f[
  x \in \mathcal{I}_i \wedge x \in \mathcal{O}_j.
\f]

## Automatic input/output identification

The only possible composition \f$r\f$ of these three functions is
\f[
  y = r(x_1, x_2) := h(f(x_1, x_2), g(f(x_1, x_2))).
\f]
The input variables of the composed function \f$r\f$ are \f$[x_1, x_2]\f$ (or their flattened concatenation), and the output variable of the composed function is simply \f$y\f$. The input/output variables are automatically identified while composing a `ComposedModel` in NEML2. This procedure is referred to as "automatic input/output identification".

In a `ComposedModel`, a _leaf_ model is a model which does not depend on any other model, and a _root_ model is a model which is not depent upon by any other model. The input variables of a `ComposedModel` is the union of the consumed variable sets \f$\cup_i \mathcal{I}_i\f$ of all the _leaf_ models, and the output variables of a `ComposedModel` is the union of the provided variable sets \f$\cup_i \mathcal{O}_i\f$ of all the _root_ models.

## Automatic dependency resolution

To evaluate the forward operator of the composed model \f$r\f$, one has to first evaluate model \f$f\f$, then model \f$g\f$, and finally model \f$h\f$. The process of sorting out such evaluation order is called "dependency resolution".

While it is possible to sort the evaluation order "by hand" for this simple example composition, it is generally not a trivial task for practical compositions with more involved dependencies. To that end, NEML2 uses [topological sort](https://en.wikipedia.org/wiki/Topological_sorting) to sort the model evaluation order, such that by the time each model is evaluated, all of its dependent models have already been evaluated.

## Automatic chain rule

Chain rule can be applied to evaluate the derivative of the forward operator with respect to the input variables, i.e.,
\f[
  \frac{\partial y}{\partial \mathbf{x}} = \left( \frac{\partial y}{\partial y_1} + \frac{\partial y}{\partial y_2} \frac{\partial y_2}{\partial y_1} \right) \frac{\partial y_1}{\partial \mathbf{x}}.
\f]
Spelling out this chain rule can be cumbersome and error-prone, especially for more complicated model compositions. The evaluation of the chain rule is automated in NEML2, and the user is only responsible for implementing the partial derivatives of each model. For example, in the implementation of `Model` \f$f\f$, the user only need to define the partial derivatives
\f[
  \frac{\partial y_1}{\partial \mathbf{x}};
\f]
similarly, `Model` \f$g\f$ only defines
\f[
  \frac{\partial y_2}{\partial y_1},
\f]
and `Model` \f$h\f$ only defines
\f[
  \frac{\partial y}{\partial y_1}, \quad \frac{\partial y}{\partial y_2}.
\f]
The assembly of the partial derivatives into the total derivative \f$\partial y / \partial \mathbf{x}\f$ using the chain rule is handled by NEML2.
