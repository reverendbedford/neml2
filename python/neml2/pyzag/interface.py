from pyzag import nonlinear

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
        state_axis="state_axis",
        forces_axis="forces_axis",
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

    def forward(self, state, forces):
        """Actually call th NEML2 model and return the residual and Jacobian

        Args:
            state (torch.tensor): tensor with the flattened state
            forces (torch.tensor): tensor with the flattened forces
        """
        pass
