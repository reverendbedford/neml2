# NEML2

### Nuclear Engineering Material model Library, version 2

NEML2 is an offshoot of [NEML](https://github.com/Argonne-National-Laboratory/neml), an earlier constitutive modeling code developed at Argonne National Laboratory.
Like NEML, NEML2 provides a flexible, modular way to build constitutive models from smaller blocks.
Unlike NEML, NEML2 vectorizes the constitutive update to efficiently run on GPUs.  NEML2 is built on top of [pytorch](https://pytorch.org/cppdocs/)
to provide GPU support, but this also means that NEML2 models have all the features of a pytorch module.  So, for example, users can take derivatives of the model
with respect to parameters using pytorch AD.

NEML2 is provided as open source software under a MIT [license](https://raw.githubusercontent.com/reverendbedford/neml2/main/LICENSE).

- - -

**NEML2 aims to provide:**

### Modular constitutive models

NEML material models are modular – they are built up from smaller pieces into a complete model. For example, a model might piece together a temperature-dependent elasticity model, a yield surface, a flow rule, and several hardening rules. Each of these submodels is independent of the other objects so that, for example, switching from conventional \f$J_2\f$ plasticity to a non \f$J_2\f$ theory requires only a one line change in an input file, if the model is already implemented, or a relatively small amount of coding to add the new yield surface if it has not been implemented. All of these objects are interchangeable. For example, the damage, viscoplastic, and rate-independent plasticity models all use the same yield (flow) surfaces, hardening rules, elasticity models, and so on.

### Extensible constitutive models

The library is structured so that adding a new feature to an existing material model should be as simple as possible and require as little code as possible. As part of this philosophy, the library only requires new components provide a few partial derivatives and NEML uses this information to assemble the Jacobian needed to do a fully implement, backward Euler integration of the ordinary differential equations comprising the model form and to provide the algorithmic tangent needed to integrate the model into an implicit finite element framework.  Moreover, in NEML2 implementations can forgo providing these partial derivatives and NEML2 will calculate them with automatic differentiation -- albeit at a significant performance cost.

### Friendly user interfaces

There are two general ways to create and interface with NEML2 material models: the python bindings and the compiled library with [HIT](https://github.com/idaholab/moose/tree/next/framework/contrib/hit) input. The python bindings are generally used for creating, fitting, and debugging new material models. In python, a material model is built up object-by-object and assembled into a complete mathematical constitutive relation. NEML2 provides several python drivers for exercising these material models in simple loading configurations. These drivers include common test types, like uniaxial tension tests and strain-controlled cyclic fatigue tests along with more esoteric drivers supporting simplified models of high temperature pressure vessels, like n-bar models and generalized plane-strain axisymmetry. NEML2 provides a full Abaqus UMAT interface and examples of how to link the compiled library into C, C++, or Fortran codes. These interfaces can be used to call NEML2 models from finite element codes. When using the compiled library, NEML2 models can be created and archived using a hierarchical HIT format.

### Strict quality assurance

NEML2 is developed under a strict quality assurance program. Because the NEML distribution does not provide full, parameterized models for any actual materials, ensuring the quality of the library is a verification problem – testing to make sure that NEML is correctly implementing the mathematical models – rather than a validation problem of comparing the results of a model to an actual test. This verification is done with extensive unit testing. This unit testing verifies every mathematical function and every derivative in the library is correctly implemented.

### CPU/GPU Vectorization

NEML2 models can be vectorized, meaning that a large batch of constitutive models can be evaluated simultaneously. The vectorized model can be evaluated both on CPU and on GPU, with a unified, intuitive user interface.

### Flexible model composition

NEML2 offers a more flexible way of composing models. Each individual model only defines the forward operator (and optionally its derivative) with a given set of inputs and outputs, without knowing anything a priori about how it is going to be used. When a set of models are *composed* together to form a composite model, dependencies among different models are automatically detected, registered, and resolved. The user has *complete control* over how NEML2 evaluates a set of models.

### Faster evaluation of chained models

As a result the dependency resolution mentioned above, an optimal order of evaluating the composed model is used to perform the forward operation -- every model in the dependency graph is evaluated once and only once, avoiding any redundant calculations.

### General implicit update

NEML2 offers a general interface for defining implicit models, unlike NEML which requires the implicit function to be in the form of an ODE.

- - -

NEML2 does not provide a database of models for any particular class of materials. There are many example materials contained in the library release, these models are included entirely for illustrative purposes and do not represent the response of any actual material.  Right now these models are solid mechanics constitutive models, providing the stress/strain response of materials.  However, NEML2 is general enough to build models of any type.
