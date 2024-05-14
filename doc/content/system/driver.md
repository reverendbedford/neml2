# Driver {#system-drivers}

[TOC]

Refer to [Syntax Documentation](@ref syntax-drivers) for the list of available objects.

Drivers are objects that "drive" the update and evolution of one or more material models and their internal data. Especially for non-autonomous material models, a driver is mandatory to evolve the mateiral model.

Drivers must derive from the base class `Driver` and override the pure virtual method `run` which returns a boolean indicating whether the model execution was successful.
