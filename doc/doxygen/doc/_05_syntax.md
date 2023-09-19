# Syntax Documentation {#syntax}

[TOC]

## AssociativeIsotropicPlasticHardening

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- equivalent_plastic_strain_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/ep_rate
- flow_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/gamma_rate
- isotropic_hardening_direction
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Nk
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [AssociativeIsotropicPlasticHardening](@ref neml2::AssociativeIsotropicPlasticHardening)

## AssociativeKinematicPlasticHardening

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- flow_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/gamma_rate
- kinematic_hardening_direction
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/NX
- kinematic_plastic_strain_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Kp_rate
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [AssociativeKinematicPlasticHardening](@ref neml2::AssociativeKinematicPlasticHardening)

## AssociativePlasticFlow

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- flow_direction
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/NM
- flow_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/gamma_rate
- plastic_strain_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Ep_rate
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [AssociativePlasticFlow](@ref neml2::AssociativePlasticFlow)

## ChabochePlasticHardening

- A
  - **Type**: Scalar
- C
  - **Type**: Scalar
- a
  - **Type**: Scalar
- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- back_stress
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/X
- flow_direction
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/NM
- flow_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/gamma_rate
- g
  - **Type**: Scalar
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [ChabochePlasticHardening](@ref neml2::ChabochePlasticHardening)

## ComposedModel

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- models
  - **Type**: vector<string>
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [ComposedModel](@ref neml2::ComposedModel)

## ElasticStrain

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- elastic_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Ee
- plastic_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Ep
- rate_form
  - **Type**: bool
  - **Default**: False
- total_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: forces/E
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [ElasticStrain](@ref neml2::ElasticStrain)

## ImplicitUpdate

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- implicit_model
  - **Type**: string
- solver
  - **Type**: string
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [ImplicitUpdate](@ref neml2::ImplicitUpdate)

## InitializedSymR2

- method
  - **Type**: string
  - **Default**: AUTO
- nbatch
  - **Type**: long
  - **Default**: -1
- values
  - **Type**: vector<Scalar>

Details: [InitializedSymR2](@ref neml2::InitializedSymR2)

## IsotropicMandelStress

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- cauchy_stress
  - **Type**: LabeledAxisAccessor
  - **Default**: state/S
- mandel_stress
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/M
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [IsotropicMandelStress](@ref neml2::IsotropicMandelStress)

## LinSpaceTensor

- end
  - **Type**: Tensor
- start
  - **Type**: Tensor
  - **Default**: 0
- steps
  - **Type**: long
  - **Default**: 0

Details: [LinSpaceTensor](@ref neml2::LinSpaceTensor)

## LinearIsotropicElasticity

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- compliance
  - **Type**: bool
  - **Default**: False
- poisson_ratio
  - **Type**: Scalar
- rate_form
  - **Type**: bool
  - **Default**: False
- strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Ee
- stress
  - **Type**: LabeledAxisAccessor
  - **Default**: state/S
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False
- youngs_modulus
  - **Type**: Scalar

Details: [LinearIsotropicElasticity](@ref neml2::LinearIsotropicElasticity)

## LinearIsotropicHardening

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- equivalent_plastic_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/ep
- hardening_modulus
  - **Type**: Scalar
- isotropic_hardening
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/k
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [LinearIsotropicHardening](@ref neml2::LinearIsotropicHardening)

## LinearKinematicHardening

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- back_stress
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/X
- hardening_modulus
  - **Type**: Scalar
- kinematic_plastic_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Kp
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [LinearKinematicHardening](@ref neml2::LinearKinematicHardening)

## LogSpaceTensor

- base
  - **Type**: double
  - **Default**: 10
- end
  - **Type**: Tensor
- start
  - **Type**: Tensor
  - **Default**: 0
- steps
  - **Type**: long
  - **Default**: 0

Details: [LogSpaceTensor](@ref neml2::LogSpaceTensor)

## ModelUnitTest

- check_AD_derivatives
  - **Type**: bool
  - **Default**: True
- check_AD_first_derivatives
  - **Type**: bool
  - **Default**: True
- check_AD_second_derivatives
  - **Type**: bool
  - **Default**: True
- check_cuda
  - **Type**: bool
  - **Default**: True
- check_first_derivatives
  - **Type**: bool
  - **Default**: True
- check_second_derivatives
  - **Type**: bool
  - **Default**: False
- derivatives_abs_tol
  - **Type**: double
  - **Default**: 1e-08
- derivatives_rel_tol
  - **Type**: double
  - **Default**: 1e-05
- input_scalar_names
  - **Type**: vector<LabeledAxisAccessor>
- input_scalar_values
  - **Type**: vector<Scalar>
- input_symr2_names
  - **Type**: vector<LabeledAxisAccessor>
- input_symr2_values
  - **Type**: vector<SymR2>
- model
  - **Type**: string
- nbatch
  - **Type**: long
  - **Default**: 1
- output_abs_tol
  - **Type**: double
  - **Default**: 1e-08
- output_rel_tol
  - **Type**: double
  - **Default**: 1e-05
- output_scalar_names
  - **Type**: vector<LabeledAxisAccessor>
- output_scalar_values
  - **Type**: vector<Scalar>
- output_symr2_names
  - **Type**: vector<LabeledAxisAccessor>
- output_symr2_values
  - **Type**: vector<SymR2>
- second_derivatives_abs_tol
  - **Type**: double
  - **Default**: 1e-08
- second_derivatives_rel_tol
  - **Type**: double
  - **Default**: 1e-05
- verbose
  - **Type**: bool
  - **Default**: False

Details: [ModelUnitTest](@ref neml2::ModelUnitTest)

## NewtonNonlinearSolver

- abs_tol
  - **Type**: double
  - **Default**: 1e-10
- max_its
  - **Type**: unsigned int
  - **Default**: 100
- rel_tol
  - **Type**: double
  - **Default**: 1e-08
- verbose
  - **Type**: bool
  - **Default**: False

Details: [NewtonNonlinearSolver](@ref neml2::NewtonNonlinearSolver)

## Normality

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- from
  - **Type**: vector<LabeledAxisAccessor>
- function
  - **Type**: LabeledAxisAccessor
- model
  - **Type**: string
- to
  - **Type**: vector<LabeledAxisAccessor>
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [Normality](@ref neml2::Normality)

## OverStress

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- back_stress
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/X
- mandel_stress
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/M
- over_stress
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/O
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [OverStress](@ref neml2::OverStress)

## PerzynaPlasticFlowRate

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- exponent
  - **Type**: Scalar
- flow_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/gamma_rate
- reference_stress
  - **Type**: Scalar
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False
- yield_function
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/fp

Details: [PerzynaPlasticFlowRate](@ref neml2::PerzynaPlasticFlowRate)

## RateIndependentPlasticFlowConstraint

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- flow_rate
  - **Type**: LabeledAxisAccessor
  - **Default**: state/gamma_rate
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False
- yield_function
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/fp

Details: [RateIndependentPlasticFlowConstraint](@ref neml2::RateIndependentPlasticFlowConstraint)

## SampleParserTestingModel

- Real
  - **Type**: double
  - **Default**: 0
- Real_vec
  - **Type**: vector<double>
- Real_vec_vec
  - **Type**: vector<vector<double>
- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- bool
  - **Type**: bool
  - **Default**: False
- bool_vec
  - **Type**: vector<bool>
- bool_vec_vec
  - **Type**: vector<vector<bool>
- int
  - **Type**: int
  - **Default**: 0
- int_vec
  - **Type**: vector<int>
- int_vec_vec
  - **Type**: vector<vector<int>
- string
  - **Type**: string
- string_vec
  - **Type**: vector<string>
- string_vec_vec
  - **Type**: vector<vector<string>
- uint
  - **Type**: unsigned int
  - **Default**: 0
- uint_vec
  - **Type**: vector<unsigned int>
- uint_vec_vec
  - **Type**: vector<vector<unsigned int>
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [SampleParserTestingModel](@ref SampleParserTestingModel)

## SampleRateModel

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [SampleRateModel](@ref SampleRateModel)

## Scalar

- batch_shape
  - **Type**: vector<long>
  - **Default**: 1
- values
  - **Type**: vector<double>

Details: [Scalar](@ref neml2::UserFixedDimTensor<neml2::Scalar>)

## ScalarBackwardEulerTimeIntegration

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- time
  - **Type**: LabeledAxisAccessor
  - **Default**: t
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False
- variable
  - **Type**: LabeledAxisAccessor

Details: [ScalarBackwardEulerTimeIntegration](@ref neml2::BackwardEulerTimeIntegration<neml2::Scalar>)

## ScalarForceRate

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- force
  - **Type**: LabeledAxisAccessor
- time
  - **Type**: LabeledAxisAccessor
  - **Default**: t
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [ScalarForceRate](@ref neml2::ForceRate<neml2::Scalar>)

## ScalarForwardEulerTimeIntegration

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- time
  - **Type**: LabeledAxisAccessor
  - **Default**: t
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False
- variable
  - **Type**: LabeledAxisAccessor

Details: [ScalarForwardEulerTimeIntegration](@ref neml2::ForwardEulerTimeIntegration<neml2::Scalar>)

## ScalarIdentityMap

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- from_var
  - **Type**: LabeledAxisAccessor
- to_var
  - **Type**: LabeledAxisAccessor
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [ScalarIdentityMap](@ref neml2::IdentityMap<neml2::Scalar>)

## ScalarSumModel

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- from_var
  - **Type**: vector<LabeledAxisAccessor>
- to_var
  - **Type**: LabeledAxisAccessor
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [ScalarSumModel](@ref neml2::SumModel<neml2::Scalar>)

## SolidMechanicsDriver

- cauchy_stress
  - **Type**: LabeledAxisAccessor
  - **Default**: forces/S
- control
  - **Type**: string
  - **Default**: STRAIN
- device
  - **Type**: string
  - **Default**: cpu
- model
  - **Type**: string
- predictor
  - **Type**: string
  - **Default**: PREVIOUS_STATE
- prescribed_strains
  - **Type**: Tensor
- prescribed_stresses
  - **Type**: Tensor
- save_as
  - **Type**: string
- show_parameters
  - **Type**: bool
  - **Default**: False
- time
  - **Type**: LabeledAxisAccessor
  - **Default**: forces/t
- times
  - **Type**: Tensor
- total_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: forces/E
- verbose
  - **Type**: bool
  - **Default**: False

Details: [SolidMechanicsDriver](@ref neml2::SolidMechanicsDriver)

## SymR2

- batch_shape
  - **Type**: vector<long>
  - **Default**: 1
- values
  - **Type**: vector<double>

Details: [SymR2](@ref neml2::UserFixedDimTensor<neml2::SymR2>)

## SymR2BackwardEulerTimeIntegration

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- time
  - **Type**: LabeledAxisAccessor
  - **Default**: t
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False
- variable
  - **Type**: LabeledAxisAccessor

Details: [SymR2BackwardEulerTimeIntegration](@ref neml2::BackwardEulerTimeIntegration<neml2::SymR2>)

## SymR2ForceRate

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- force
  - **Type**: LabeledAxisAccessor
- time
  - **Type**: LabeledAxisAccessor
  - **Default**: t
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [SymR2ForceRate](@ref neml2::ForceRate<neml2::SymR2>)

## SymR2ForwardEulerTimeIntegration

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- time
  - **Type**: LabeledAxisAccessor
  - **Default**: t
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False
- variable
  - **Type**: LabeledAxisAccessor

Details: [SymR2ForwardEulerTimeIntegration](@ref neml2::ForwardEulerTimeIntegration<neml2::SymR2>)

## SymR2IdentityMap

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- from_var
  - **Type**: LabeledAxisAccessor
- to_var
  - **Type**: LabeledAxisAccessor
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [SymR2IdentityMap](@ref neml2::IdentityMap<neml2::SymR2>)

## SymR2Invariant

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- invariant
  - **Type**: LabeledAxisAccessor
- invariant_type
  - **Type**: string
- tensor
  - **Type**: LabeledAxisAccessor
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [SymR2Invariant](@ref neml2::SymR2Invariant)

## SymR2SumModel

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- from_var
  - **Type**: vector<LabeledAxisAccessor>
- to_var
  - **Type**: LabeledAxisAccessor
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [SymR2SumModel](@ref neml2::SumModel<neml2::SymR2>)

## SymSymR4

- batch_shape
  - **Type**: vector<long>
  - **Default**: 1
- values
  - **Type**: vector<double>

Details: [SymSymR4](@ref neml2::UserFixedDimTensor<neml2::SymSymR4>)

## Tensor

- shape
  - **Type**: vector<long>
- values
  - **Type**: vector<double>

Details: [Tensor](@ref neml2::UserTensor)

## TotalStrain

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- elastic_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Ee
- plastic_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/Ep
- rate_form
  - **Type**: bool
  - **Default**: False
- total_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/E
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [TotalStrain](@ref neml2::TotalStrain)

## TransientRegression

- atol
  - **Type**: double
  - **Default**: 1e-08
- driver
  - **Type**: string
- reference
  - **Type**: string
- rtol
  - **Type**: double
  - **Default**: 1e-05
- verbose
  - **Type**: bool
  - **Default**: False

Details: [TransientRegression](@ref neml2::TransientRegression)

## VTestTimeSeries

- variable
  - **Type**: string
- variable_type
  - **Type**: string
- vtest
  - **Type**: string

Details: [VTestTimeSeries](@ref neml2::VTestTimeSeries)

## VTestVerification

- atol
  - **Type**: double
  - **Default**: 1e-08
- driver
  - **Type**: string
- references
  - **Type**: vector<Tensor>
- rtol
  - **Type**: double
  - **Default**: 1e-05
- variables
  - **Type**: vector<string>
- verbose
  - **Type**: bool
  - **Default**: False

Details: [VTestVerification](@ref neml2::VTestVerification)

## VoceIsotropicHardening

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- equivalent_plastic_strain
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/ep
- isotropic_hardening
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/k
- saturated_hardening
  - **Type**: Scalar
- saturation_rate
  - **Type**: Scalar
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False

Details: [VoceIsotropicHardening](@ref neml2::VoceIsotropicHardening)

## YieldFunction

- additional_outputs
  - **Type**: vector<LabeledAxisAccessor>
- isotropic_hardening
  - **Type**: LabeledAxisAccessor
- stress_measure
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/sm
- use_AD_first_derivative
  - **Type**: bool
  - **Default**: False
- use_AD_second_derivative
  - **Type**: bool
  - **Default**: False
- yield_function
  - **Type**: LabeledAxisAccessor
  - **Default**: state/internal/fp
- yield_stress
  - **Type**: Scalar

Details: [YieldFunction](@ref neml2::YieldFunction)

