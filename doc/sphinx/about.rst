About NEML
==========

What NEML is
------------

NEML (the Nuclear Engineering Material model Library) is a tool for creating
and running structural material models.
While it was originally developed to model high temperature nuclear reactors,
the tool is general enough to apply to most types of structural materials.

The focus of NEML is on modularity and extensibility.
The library is structured so that adding a new feature to an existing material
model should be as simple as possible and require as little code as possible.

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

There are two general ways to create and interface with NEML material models:
the python bindings and the compiled library with XML input.
The python bindings are generally used for creating, fitting, and debugging
new material models.
In python, a material model is built up object-by-object and assembled into
a complete mathematical constitutive relation.
NEML provides several python drivers for exercising these material models in
simple loading configurations.
These drivers include common test types, like uniaxial tension tests and
strain-controlled cyclic fatigue tests along with more esoteric drivers
supporting simplified models of high temperature pressure vessels, like
*n*-bar models and generalized plane-strain axisymmetry.
NEML provides a full Abaqus UMAT interface and examples of how to link the
compiled library into C, C++, or Fortran codes.
These interfaces can be used to call NEML models from finite element
codes.
When using the compiled library, NEML models can be created and archived
using a hierarchical XML format.

NEML is developed under a strict quality assurance program.  Because, as
discussed below, the NEML distribution does not provide models for any
actual materials, ensuring the quality of the library is a verification 
problem -- testing to make sure that NEML is correctly implementing the
mathematical models -- rather than a validation problem of comparing the
results of a model to an actual test.
This verification is done with extensive unit testing through the python
interface.
This unit testing verifies every mathematical function and every derivative
in the library is correctly implemented. 


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

.. _conventions:

Mathematical conventions
------------------------

Mandel notation
^^^^^^^^^^^^^^^

NEML models work in three dimensions.
This means second order tensors can be expressed by 3-by-3 arrays and
fourth order tensors are 3-by-3-by-3-by-3 arrays.
This three dimensional interface can naturally accommodate 3D and 2D 
plane strain stress updates.
A python example demonstrates how to use the 3D interface to degenerate
the models to the standard strain-controlled uniaxial material interface where
the stress in the loading direction is strain controlled and all the
remaining stress components are stress controlled to zero stress.

NEML uses the Mandel notation to convert symmetric second and fourth order
tensors to vectors and matrices.
The convention transforms the second order tensor

.. math::

      \left[\begin{array}{ccc}
      \sigma_{11} & \sigma_{12} & \sigma_{13}\\
      \sigma_{12} & \sigma_{22} & \sigma_{23}\\
      \sigma_{13} & \sigma_{23} & \sigma_{33}
      \end{array}\right]
      \rightarrow
      \left[\begin{array}{cccccc}
      \sigma_{11} & \sigma_{22} & \sigma_{33} & \sqrt{2}\sigma_{23} & 
      \sqrt{2}\sigma_{13} & \sqrt{2}\sigma_{12}\end{array}\right]

and, after transformation, a fourth order tensor :math:`\mathbf{\mathfrak{C}}` becomes

.. math::

      \left[\begin{array}{cccccc}
      C_{1111} & C_{1122} & C_{1133} & \sqrt{2}C_{1123} & \sqrt{2}C_{1113} & \sqrt{2}C_{1112}\\
      C_{1122} & C_{2222} & C_{2233} & \sqrt{2}C_{2223} & \sqrt{2}C_{2213} & \sqrt{2}C_{2212}\\
      C_{1133} & C_{2233} & C_{3333} & \sqrt{2}C_{3323} & \sqrt{2}C_{3313} & \sqrt{2}C_{3312}\\
      \sqrt{2}C_{1123} & \sqrt{2}C_{2223} & \sqrt{2}C_{3323} & 2C_{2323} & 2C_{2313} & 2C_{2312}\\
      \sqrt{2}C_{1113} & \sqrt{2}C_{2213} & \sqrt{2}C_{3313} & 2C_{2313} & 2C_{1313} & 2C_{1312}\\
      \sqrt{2}C_{1112} & \sqrt{2}C_{2212} & \sqrt{2}C_{3312} & 2C_{2312} & 2C_{1312} & 2C_{1212}
      \end{array}\right].

For symmetric two second order tensors :math:`\mathbf{A}` and :math:`\mathbf{B}`
and their Mandel vectors :math:`\hat{\mathbf{a}}` and :math:`\hat{\mathbf{b}}`
the relation 

.. math::

      \mathbf{A}:\mathbf{B}=\hat{\mathbf{a}}\cdot\hat{\mathbf{b}}

expresses the utility of this convention.
Similarly, given the symmetric fourth order tensor :math:`\mathbf{\mathfrak{C}}`
and its equivalent Mandel matrix :math:`\hat{\mathbf{C}}`
contraction over two adjacent indices

.. math::

      \mathbf{A}=\mathbf{\mathfrak{C}}:\mathbf{B}

simply becomes matrix-vector multiplication

.. math::

      \hat{\mathbf{a}}=\hat{\mathbf{C}}\cdot\hat{\mathbf{b}}.

The Mandel convention is relatively uncommon in finite element software, and so
the user must be careful to convert back and forth from the Mandel convention to
whichever convention the calling software uses.
The Abaqus UMAT interface provided with NEML demonstrates how to make this
conversion before and after each call.

Typographically, the manual uses a lower case, standard font (:math:`x`) to
represent a scalar, a bold lower case to represent a vector
(:math:`\mathbf{x}`), a bold upper case to represent a second order tensor
(:math:`\mathbf{X}`), and a bold upper case Fraktur to represent a fourth
order tensor (:math:`\mathbf{\mathfrak{X}}`).
There are certain exceptions to the upper case/lower case convention for
commonly used notation.
For example the stress and strain tensors are denoted :math:`\bm{\sigma}` and
:math:`\bm{\varepsilon}`, respectively.
Internally in NEML all tensor operations are implemented as Mandel dot
products.
However, the documentation writes the full tensor form of the equations.

Throughout the documentation the deviator of a second order tensor is 
denoted:

.. math::
   \operatorname{dev}\left(\mathbf{X}\right) = \mathbf{X} - \frac{1}{3} 
      \operatorname{tr}\left(\mathbf{X}\right) \mathbf{I}

with :math:`\operatorname{tr}` the trace and :math:`\mathbf{I}` the
identity tensor.

When describing collections of objects the manual uses square brackets.
For example,

.. math::
   \left[ \begin{array}{ccc} s & \mathbf{X} & \mathbf{v} \end{array}\right]

Indicates a collection of a scalar :math:`s`, a vector representing a
second order tensor :math:`\mathbf{x}` in Mandel notation, and a 
vector :math:`\mathbf{v}`.
These collections are ordered.
This notation indicates the implementation is concatenating the quantities
into a flat, 1D array (in this case with length :math:`1 + 6 + 3 = 10`).

Interfaces
^^^^^^^^^^

The documentation describes NEML as a collection of interfaces. 
An interface is a collection of functions with the same inputs but
different outputs.
These interfaces are implemented as C++ objects in NEML.
The documentation describes an interface with the notation:

.. math::

   a, \mathbf{B}, \mathbf{\mathfrak{C}} \leftarrow \mathcal{F} \left(d, \mathbf{e}, \mathbf{F} \right)

This is an interface that takes a scalar :math:`d`, vector :math:`\mathbf{e}`,
and second order tensor :math:`\mathbf{F}` as input and returns a scalar
:math:`a`, second order tensor :math:`\mathbf{B}`, and fourth order symmetric
tensor :math:`\mathbf{\mathfrak{C}}` as output.
The interface might be implemented as three individual functions

.. math::
   a = f \left(d, \mathbf{e}, \mathbf{F} \right)

   \mathbf{B} = \mathbf{F} \left(d, \mathbf{e}, \mathbf{F} \right)

   \mathbf{\mathfrak{C}} = \mathbf{\mathfrak{F}} \left(d, \mathbf{e}, \mathbf{F} \right).



