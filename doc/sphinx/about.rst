About NEML2
===========

NEML2 is the offshoot of NEML, an earlier code developed at ANL. NEML2 differs
from its predecessor in three important ways:

* Vectorization: NEML2 models can be vectorized, meaning that a large batch of
  constitutive models can be evaluated simultaneously. The vectorized model can
  utilize both CPU and GPU, with a unified, intuitive user interface.
* Model composition: NEML2 offers a more flexible way of composing models. Each
  individual model only defines the forward operator (and optionally its
  derivative) with a given set of inputs and outputs, without know anything a
  priori about how it is going to be used. The user has *total control* of how
  NEML2 evaluates a set of models.
* Implicit update: NEML2 offers a general interface for defining implicit models,
  unlike NEML2 which requires the implicit function to be in the form of an ODE.


What NEML is
------------

      This paragraph is shamelessly taken from NEML, which also does a
      good job at describing NEML2.

NEML (the Nuclear Engineering Material model Library) is a tool for creating
and running structural material models.
While it was originally developed to model high temperature nuclear reactors,
the tool is general enough to apply to most types of structural materials.

The focus of NEML is on modularity and extensibility.
The library is structured so that adding a new feature to an existing material
model should be as simple as possible and requires as little code as possible.

NEML material models are modular -- they are built up from smaller pieces into
a complete model.
For example, a model might piece together a temperature-dependent elasticity
model, a yield surface, a flow rule, and several hardening rules.
Each of these submodels is independent of the other objects
so that, for example, switching from conventional :math:`J_2` plasticity
to a non-:math:`J_2` theory requires only a one line change in an input file,
if the model is already implemented, or a relatively small amount of coding
to add the new yield surface if it has not been implemented.
All of these objects are interchangeable.
For example, the damage, viscoplastic, and rate-independent plasticity
models all use the same yield (flow) surfaces, hardening rules, elasticity
models, and so on.

As part of this philosophy, the library only requires new components
provide a few partial derivatives and NEML uses this information to assemble
the Jacobian needed to do a fully implement, backward Euler integration of the
ordinary differential equations comprising the model form and to provide
the algorithmic tangent needed to integrate the model into an implicit
finite element framework.

What NEML is not
----------------

NEML does not provide a database of models for any particular class of
materials.
There are many example materials contained in the library release, these
models are included entirely for illustrative purposes and do not
represent the response of any actual material.

NEML will not be the fastest constitutive model when call from an external
FE program.
The focus of the library is on extensibility, rather than computational
efficiency.

Interface with NEML2
--------------------

Placeholder. Well, at least NEML2 can be interfaced with C or C++, and we
plan to add model (de)serialization support. Python interface is also on
the roadmap.

Quality assurance
-----------------

NEML is developed under a strict quality assurance program.  Because, as
discussed below, the NEML distribution does not provide models for any
actual materials, ensuring the quality of the library is a verification
problem -- testing to make sure that NEML is correctly implementing the
mathematical models -- rather than a validation problem of comparing the
results of a model to an actual test.
This verification is done with extensive unit testing using Catch2. This unit
testing verifies every mathematical function and every derivative
in the library is correctly implemented.


