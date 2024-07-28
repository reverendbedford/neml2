# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import graphviz

subaxes = ["state", "old_state", "forces", "old_forces", "residual", "parameters"]

global_attributes = {
    "fontname": "Arial",
    "fontsize": "24pt",
    "ranksep": "1",
    "peripheries": "0",
}
global_node_attributes = {
    "shape": "box",
    "fontname": "Courier New",
    "fontsize": "24pt",
}
submodel_attributes = {
    "bgcolor": "gray95",
}
submodel_node_attributes = {
    "fontsize": "18pt",
}
submodel_name_node_attributes = {
    "fontsize": "36pt",
    "color": "#003ec9",
    "fillcolor": "#003ec935",
    "style": "bold,filled",
}
input_subaxis_attributes = {
    "peripheries": "",
    "style": "rounded,dashed",
    "labelloc": "t",
}
input_subaxis_node_attributes = {}
output_subaxis_attributes = {
    "peripheries": "",
    "style": "rounded,dashed",
    "labelloc": "b",
}
output_subaxis_node_attributes = {}
input_node_attributes = {
    "style": "filled",
    "color": "gray60",
    "fillcolor": "gray90",
}
output_node_attributes = {
    "style": "filled",
    "color": "#00540150",
    "fillcolor": "#00540115",
}
input_edge_attributes = {
    "arrowsize": "2",
    "minlen": "1",
}
output_edge_attributes = {
    "arrowsize": "2",
    "minlen": "1",
}
source_edge_attributes = {
    "arrowsize": "2",
    "minlen": "2.5",
}
sink_edge_attributes = {
    "style": "dashed",
    "arrowsize": "2",
    "minlen": "2.5",
}


def render(model, *args, **kwargs):
    """
    Render a model's composition dependency graph in graphviz format.
    Arguments and keyword arguments are passed to graphviz.Digraph.render.

    :param model: Model to render
    """
    g = graphviz.Digraph(name="parent")
    _set_global(g)
    _add_model(g, model)
    g.render(*args, **kwargs)


def _set_global(graph):
    graph.attr(**global_attributes)
    graph.node_attr.update(global_node_attributes)


def _add_model(graph, model):
    _add_variables(graph, model, True)
    _add_variables(graph, model, False)

    for _, submodel in model.named_submodels().items():
        _add_submodel(graph, submodel)
        _link_variables(graph, submodel, True)
        _link_variables(graph, submodel, False)
        _link_output_variables(graph, submodel, model)


def _add_submodel(graph, model):
    cname = "cluster_{}".format(model.name)
    with graph.subgraph(name=cname) as mgraph:
        mgraph.attr(**submodel_attributes)
        mgraph.node_attr.update(submodel_node_attributes)
        # add model name
        mgraph.node(
            model.name,
            label=model.name + "\\n[{}]".format(model.type),
            **submodel_name_node_attributes,
        )
        # add output variables
        _add_variables(mgraph, model, False)


def _add_variables(graph, model, input):
    axis = model.input_axis() if input else model.output_axis()
    xname = "input" if input else "output"
    deps = model.dependency()
    cname = "cluster_{}_{}".format(model.name, xname)
    with graph.subgraph(name=cname) as vgraph:
        for subaxis in subaxes:
            if axis.has_subaxis(subaxis):
                cname = "cluster_{}_{}_{}".format(model.name, xname, subaxis)
                with vgraph.subgraph(name=cname) as s:
                    s.attr(
                        label=subaxis,
                        **(
                            input_subaxis_attributes
                            if input
                            else output_subaxis_attributes
                        ),
                    )
                    s.node_attr.update(
                        input_subaxis_node_attributes
                        if input
                        else output_subaxis_node_attributes
                    )
                    for var in axis.subaxis(subaxis).variable_names():
                        vname = "{}/{}".format(subaxis, var)
                        if input and vname in deps:
                            continue
                        vlabel = "{}_{}_{}".format(model.name, xname, vname)
                        vtype = str(
                            model.input_type(subaxis + "/" + var)
                            if input
                            else model.output_type(subaxis + "/" + var)
                        ).split(".")[-1]
                        s.node(
                            name=vlabel,
                            label="{}\\n[{}]".format(var, vtype),
                            **(
                                input_node_attributes
                                if input
                                else output_node_attributes
                            ),
                        )


def _link_variables(graph, model, input):
    axis = model.input_axis() if input else model.output_axis()
    deps = model.dependency()
    for var in axis.variable_names():
        if input:
            if var in deps:
                dep = deps[var]
                mname = dep.name
                xname = "input" if model.name in dep.named_submodels() else "output"
                vname = "{}_{}_{}".format(mname, xname, var)
                graph.edge(vname, model.name, **source_edge_attributes)
            else:
                mname = model.name
                xname = "input"
                vname = "{}_{}_{}".format(mname, xname, var)
                graph.edge(vname, model.name, **input_edge_attributes)
        else:
            mname = model.name
            xname = "output"
            vname = "{}_{}_{}".format(mname, xname, var)
            graph.edge(model.name, vname, **input_edge_attributes)


def _link_output_variables(graph, submodel, model):
    ovars = model.output_axis().variable_names()
    v = "{}_output_{}"
    for ovar in submodel.output_axis().variable_names():
        if ovar in ovars:
            graph.edge(
                v.format(submodel.name, ovar),
                v.format(model.name, ovar),
                **sink_edge_attributes,
            )
