from pyzag import nonlinear

from neml2.tensors import Tensor
from neml2.tensors import LabeledAxisAccessor as AA

import torch


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
                    p.data,
                    Tensor(
                        self.model.get_parameter(self.parameter_name_map[n])
                    ).batch.dim(),
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

    @property
    def nstate(self):
        return self.model.input_axis().subaxis("state").storage_size()

    @property
    def nforce(self):
        return self.model.input_axis().subaxis("forces").storage_size()

    def forward(self, state, forces):
        """Actually call th NEML2 model and return the residual and Jacobian

        Args:
            state (torch.tensor): tensor with the flattened state
            forces (torch.tensor): tensor with the flattened forces
        """
        pass
