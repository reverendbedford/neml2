# Mathematical Conventions {#math}

[TOC]

## Typography

The manual uses a lower case, standard font \f$x\f$ to represent a scalar, a bold lower case to represent a vector
\f$\mathbf{x}\f$, a bold upper case to represent a second order tensor \f$\mathbf{x}\f$, and a bold upper case Fraktur to represent a fourth order tensor \f$\mathbf{\mathfrak{X}}\f$.

There are certain exceptions to the upper case/lower case convention for commonly used notation. For example the stress and strain tensors are denoted \f$\mathbf{\sigma}\f$ and \f$\mathbf{\varepsilon}\f$, respectively.

Internally in NEML2 all tensor operations are implemented as [Mandel](#mandel) dot products. However, the documentation writes the full tensor form of the equations.

NEML2 models work in three dimensions. This means second order tensors can be expressed by 3-by-3 arrays and
fourth order tensors are 3-by-3-by-3-by-3 arrays.

## Mandel notation {#mandel}

NEML2 uses the Mandel notation to convert symmetric second and fourth order tensors to vectors and matrices.
The convention transforms the second order tensor
\f[
      \left[\begin{array}{ccc}
      \sigma_{11} & \sigma_{12} & \sigma_{13}\\
      \sigma_{12} & \sigma_{22} & \sigma_{23}\\
      \sigma_{13} & \sigma_{23} & \sigma_{33}
      \end{array}\right]
\f]
to
\f[
      \left[\begin{array}{cccccc}
      \sigma_{11} & \sigma_{22} & \sigma_{33} & \sqrt{2}\sigma_{23} &
      \sqrt{2}\sigma_{13} & \sqrt{2}\sigma_{12}\end{array}\right]
\f]
and, after transformation, a fourth order tensor \f$\mathbf{\mathfrak{C}}\f$ becomes
\f[
      \left[\begin{array}{cccccc}
      C_{1111} & C_{1122} & C_{1133} & \sqrt{2}C_{1123} & \sqrt{2}C_{1113} & \sqrt{2}C_{1112}\\
      C_{1122} & C_{2222} & C_{2233} & \sqrt{2}C_{2223} & \sqrt{2}C_{2213} & \sqrt{2}C_{2212}\\
      C_{1133} & C_{2233} & C_{3333} & \sqrt{2}C_{3323} & \sqrt{2}C_{3313} & \sqrt{2}C_{3312}\\
      \sqrt{2}C_{1123} & \sqrt{2}C_{2223} & \sqrt{2}C_{3323} & 2C_{2323} & 2C_{2313} & 2C_{2312}\\
      \sqrt{2}C_{1113} & \sqrt{2}C_{2213} & \sqrt{2}C_{3313} & 2C_{2313} & 2C_{1313} & 2C_{1312}\\
      \sqrt{2}C_{1112} & \sqrt{2}C_{2212} & \sqrt{2}C_{3312} & 2C_{2312} & 2C_{1312} & 2C_{1212}
      \end{array}\right].
\f]
The inner product of two symmetric second order tensors \f$\mathbf{A}\f$ and \f$\mathbf{B}\f$ can be conveniently expressed in their Mandel notations \f$\hat{\mathbf{a}}\f$ and \f$\hat{\mathbf{b}}\f$ as
\f[
      \mathbf{A}:\mathbf{B}=\hat{\mathbf{a}}\cdot\hat{\mathbf{b}}
\f]

which expresses the utility of this convention.

Similarly, given the symmetric fourth order tensor \f$\mathbf{\mathfrak{C}}\f$ and its equivalent Mandel matrix \f$\hat{\mathbf{C}}\f$ contraction over two adjacent indices
\f[
      \mathbf{A}=\mathbf{\mathfrak{C}}:\mathbf{B}
\f]
simply becomes matrix-vector multiplication
\f[
      \hat{\mathbf{a}}=\hat{\mathbf{C}}\cdot\hat{\mathbf{b}}.
\f]

The Mandel convention is relatively uncommon in finite element software, and so the user must be careful to convert back and forth from the Mandel convention to whichever convention the calling software uses. The Abaqus UMAT interface provided with NEML2 demonstrates how to make this conversion before and after each call.

## Commonly used operators

The deviator of a second order tensor is denoted:
\f[
   \operatorname{dev}\left(\mathbf{X}\right) = \mathbf{X} - \frac{1}{3}
      \operatorname{tr}\left(\mathbf{X}\right) \mathbf{I}
\f]
with \f$\operatorname{tr}\f$ the trace and \f$\mathbf{I}\f$ the identity tensor.

## Concatenation of flattened quantities

When describing collections of objects the manual uses square brackets. For example,
\f[
   \left[ \begin{array}{ccc} s & \mathbf{X} & \mathbf{v} \end{array}\right]
\f]
indicates a collection of a scalar \f$s\f$, a vector representing a second order tensor \f$\mathbf{X}\f$ in Mandel notation, and a vector \f$\mathbf{v}\f$. This notation indicates the implementation is concatenating the quantities into a flat, 1D array (in this case with length \f$1 + 6 + 3 = 10\f$) preserving the original order.

