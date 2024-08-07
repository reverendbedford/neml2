from pyzag import nonlinear

import neml2
from neml2.tensors import Tensor
from neml2.tensors import LabeledAxisAccessor as AA

import torch


def assemble_vector(axis, tensors):
    """Assemble a LabeledVector from a collection of tensors

    Args:
        axis (LabeledAxis): axis to use to setup LabeledVector
        tensor (dict of torch.tensor): dictionary mapping names to tensors
    """
    random_tensor = next(iter(tensors.values()))
    batch_shape = random_tensor.shape[:-1]
    device = random_tensor.device

    vector = neml2.LabeledVector.zeros(batch_shape, [axis], device=device)

    for name, value in tensors.items():
        vector.base[name] = Tensor(value, len(batch_shape))

    return vector


class NEML2PyzagModel(nonlinear.NonlinearRecursiveFunction):
    """Wraps a NEML2 model into a `nonlinear.NonlinearRecursiveFunction`

    Args:
        model (NEML2 model): the model to wrap

    Keyword Args:
        exclude_parameters (list of str): exclude these parameters from being wrapped as a pytorch parameter
        state_axis (str): name of the state axis
        forces_axis (str): name of the forces axis
        residual_axis (str): name of the residual axis
        old_prefix (str): prefix on the name of an axis to get the old values
        neml2_sep_char (str): parameter seperator character used in NEML2
        python_sep_char (str): seperator character to use in python used to name parameters
    """

    def __init__(
        self,
        model,
        exclude_parameters=[],
        state_axis="state",
        forces_axis="forces",
        residual_axis="residual",
        old_prefix="old_",
        neml2_sep_char=".",
        python_sep_char="_",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model = model
        self.lookback = (
            1  # Hard coded because there aren't any other options in NEML2 right now
        )

        self.neml2_sep_char = neml2_sep_char
        self.python_sep_char = python_sep_char
        self._setup_parameters(exclude_parameters)

        self.state_axis = state_axis
        self.forces_axis = forces_axis
        self.residual_axis = residual_axis
        self.old_prefix = old_prefix

        self._check_model()

    def _setup_parameters(self, exclude_parameters):
        """Mirror parameters of the neml model with torch.nn.Parameter

        Args:
            exclude_parameters (list of str): neml parameters to exclude
        """
        self.parameter_name_map = {}
        for neml_name, neml_param in self.model.named_parameters().items():
            if neml_name in exclude_parameters:
                continue
            neml_param.requires_grad_(True)
            rename = neml_name.replace(self.neml2_sep_char, self.python_sep_char)
            self.parameter_name_map[rename] = neml_name
            self.register_parameter(rename, torch.nn.Parameter(neml_param.torch()))

    def _update_parameter_values(self):
        """Copy over new parameter values"""
        self.model.set_parameters(
            {
                self.parameter_name_map[n]: Tensor(
                    p,
                    self.model.get_parameter(self.parameter_name_map[n])
                    .tensor()
                    .batch.dim(),
                )
                for n, p in self.named_parameters()
            }
        )

    def _check_model(self):
        """Simple consistency checks, could be a debug check but we only call this once"""
        should_axes = [
            self.state_axis,
            self.old_prefix + self.state_axis,
            self.forces_axis,
            self.old_prefix + self.forces_axis,
        ]
        if self.model.input_axis().nsubaxis() != len(should_axes):
            raise ValueError(
                "Wrapped NEML2 model should only have 4 subaxes on the input axis"
            )
        for axis in should_axes:
            if not self.model.input_axis().has_subaxis(AA(axis)):
                raise ValueError("Wrapped NEML2 model missing input subaxis %s" % axis)

        # Output axis should just have the residual
        if self.model.output_axis().nsubaxis() != 1:
            raise ValueError(
                "Wrapped NEML2 model should only have 1 subaxes on the output axis"
            )

        if not self.model.output_axis().has_subaxis(AA(self.residual_axis)):
            raise ValueError(
                "Wrapped NEML2 model is missing required output subaxis %s"
                % self.residual_axis
            )

        # And all the variables on state should match the variables in the residual
        for name in self.model.input_axis().subaxis(self.state_axis).variable_names():
            self.model.output_axis().subaxis(self.residual_axis).has_variable(AA(name))

        # Everything in old_state should be in state (but not the other way around)
        for name in (
            self.model.input_axis()
            .subaxis(self.old_prefix + self.state_axis)
            .variable_names()
        ):
            if not self.model.input_axis().subaxis(self.state_axis).has_variable(name):
                raise ValueError(
                    "State variable %s is in old state but not in state" % name
                )

        # Everything in old_forces should be in forces (but not the other way around)
        for name in (
            self.model.input_axis()
            .subaxis(self.old_prefix + self.forces_axis)
            .variable_names()
        ):
            if not self.model.input_axis().subaxis(self.forces_axis).has_variable(name):
                raise ValueError(
                    "Force variable %s is in old forces but not in forces" % name
                )

    @property
    def nstate(self):
        return self.model.input_axis().subaxis("state").storage_size()

    @property
    def nforce(self):
        return self.model.input_axis().subaxis("forces").storage_size()

    def collect_forces(self, tensor_dict):
        """Assemble the forces from a dictionary of tensors

        Args:
            tensor_dict (dict of tensors): dictionary of tensors containing the forces
        """
        return assemble_vector(
            self.model.input_axis().subaxis(self.forces_axis), tensor_dict
        ).torch()

    def collect_state(self, tensor_dict):
        """Assemble the state from a dictionary of tensors

        Args:
            tensor_dict (dict of tensors): dictionary of tensors containing the forces
        """
        return assemble_vector(
            self.model.input_axis().subaxis(self.state_axis), tensor_dict
        ).torch()

    def _reduce_axis(self, reduce_axis, full_axis, full_tensor):
        """Reduce a tensor spanning full_axis to only the vars on reduce_axis

        Args:
            reduce_axis (LabeledAxis): reduced set of variables
            full_axis (LabeledAxis): full set of variables
            full_tensor (torch.tensor): tensor representing the full set of variables
        """

        batch_shape = full_tensor.shape[:-1]
        full = neml2.LabeledVector(full_tensor, [full_axis])
        reduced = neml2.LabeledVector.zeros(
            batch_shape, [reduce_axis], device=full_tensor.device
        )
        reduced.fill(full)

        return reduced.tensor()

    def _assemble_input(self, state, forces):
        """Assemble the model input from the flat tensors

        Args:
            state (torch.tensor): tensor containing the model state
            forces (torch.tensor): tensor containing the model forces
        """
        batch_shape = (state.shape[0] - self.lookback,) + state.shape[1:-1]
        bdim = len(batch_shape)

        input = neml2.LabeledVector.zeros(
            batch_shape, [self.model.input_axis()], device=state.device
        )

        input.base[self.state_axis] = Tensor(state[self.lookback :], bdim)
        # This deals with variables not in old_state
        input.base[self.old_prefix + self.state_axis] = self._reduce_axis(
            self.model.input_axis().subaxis(self.old_prefix + self.state_axis),
            self.model.input_axis().subaxis(self.state_axis),
            state[: -self.lookback],
        )
        input.base[self.forces_axis] = Tensor(forces[self.lookback :], bdim)
        # This deals with variables not in old_forces
        input.base[self.old_prefix + self.forces_axis] = self._reduce_axis(
            self.model.input_axis().subaxis(self.old_prefix + self.forces_axis),
            self.model.input_axis().subaxis(self.forces_axis),
            forces[: -self.lookback],
        )

        self.model.reinit(input.batch.shape, 1)

        return input

    def _extract_jacobian(self, J):
        """Extract the Jacobian components we need from the NEML output

        Args:
            J (LabeledMatrix): full jacobian from the NEML model
        """
        J_new = J.base[self.residual_axis, self.state_axis].torch()
        # Now we need to pad the variables not in old_state with zeros
        J_old_reduced = neml2.tensors.LabeledMatrix(
            J.base[self.residual_axis, self.old_prefix + self.state_axis],
            [
                self.model.output_axis().subaxis(self.residual_axis),
                self.model.input_axis().subaxis(self.old_prefix + self.state_axis),
            ],
        )
        J_old_full = neml2.tensors.LabeledMatrix.zeros(
            J_new.shape[:-2],
            [
                self.model.output_axis().subaxis(self.residual_axis),
                self.model.input_axis().subaxis(self.state_axis),
            ],
            device=J_new.device,
        )
        J_old_full.fill(J_old_reduced, common_first=True)

        return torch.stack([J_old_full.torch(), J_new])

    def forward(self, state, forces):
        """Actually call the NEML2 model and return the residual and Jacobian

        Args:
            state (torch.tensor): tensor with the flattened state
            forces (torch.tensor): tensor with the flattened forces

        The helper methods `collect_forces` and `collect_state` can be used to
        assemble individual tensors into the flattened state and forces tensor
        """
        # Make a big LabeledVector with the input
        x = self._assemble_input(state, forces)

        # Update the parameter values
        self._update_parameter_values()

        # Call the model
        y, J = self.model.value_and_dvalue(x)

        return y.torch(), self._extract_jacobian(J)
