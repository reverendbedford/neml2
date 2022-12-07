Mathematical conventions
========================

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


