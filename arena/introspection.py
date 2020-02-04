from enum import Enum
from typing import NamedTuple, List, Dict, Set, Optional, Callable, Tuple
from collections import deque

import torch
from torch import nn


#
# Inspired by torchviz and pytorch-summary
#


class ControlFlow(Enum):
    CONTINUE = 1
    STOP = 2
    BREAK = 3


class GradId(NamedTuple):
    id: str

    def __repr__(self):
        return f'grad-{self.id}'


class TensorId(NamedTuple):
    id: str

    def __repr__(self):
        return f'tensor-{self.id}'


class NodeData(NamedTuple):
    variable_id: Optional[TensorId]
    variable_size: Optional[List[int]]
    param_name: Optional[str]
    class_name: str

    def __repr__(self):
        dump = f'{self.class_name}'

        if self.variable_id is not None:
            dump += f' - {self.variable_id}'

        if self.variable_size is not None:
            dump += ' (' + ', '.join(['%d' % v for v in self.variable_size]) + ')'

        if self.param_name is not None:
            dump += f' # {self.param_name}'

        return dump


class Beam(NamedTuple):
    seed: GradId
    inners: Set[GradId]
    terminators: Set[GradId]
    edges: List[Tuple[GradId, GradId]]

    def __repr__(self):
        dump = ''

        dump += f'seed: {self.seed}\n'
        dump += f'inners: {", ".join([str(gid) for gid in self.inners])}\n'
        dump += f'terminators: {", ".join([str(gid) for gid in self.terminators])}\n'
        dump += f'edges: {", ".join([str(a) + " <- " + str(b) for a, b in self.edges])}\n'

        return dump


def grad_id(o) -> GradId:
    return GradId(hex(id(o)))


def tensor_id(o) -> TensorId:
    return TensorId(hex(id(o)))


class NetworkGraph(object):
    def __init__(self):
        self.nodes: Dict[GradId, NodeData] = {}     # id -> data
        self.edges: Dict[GradId, Set[GradId]] = {}  # out -> set(in)

    def add_node(self, gid: GradId, data: NodeData):
        self.nodes[gid] = data

    def add_edge(self, gid_in: GradId, gid_out: GradId):
        assert gid_in in self.nodes
        assert gid_out in self.nodes

        gids_in = self.edges.setdefault(gid_out, set())
        gids_in.add(gid_in)

    def contains(self, gid: GradId) -> bool:
        return gid in self.nodes

    def beam_search(self, gid: GradId, control: Callable[[GradId, NodeData], ControlFlow]) -> Optional[Beam]:
        assert gid in self.nodes

        stack = deque([gid])

        inners = set()
        terminators = set()
        beam_edges = []

        while len(stack) > 0:
            gid_out = stack.pop()
            gids_in = self.edges.get(gid_out) or set()

            if len(gids_in) == 0:
                return None

            for gid_in in gids_in:
                beam_edges.append((gid_out, gid_in))

                flow = control(gid_in, self.nodes[gid_in])
                if flow == ControlFlow.STOP:
                    terminators.add(gid_in)
                elif flow == ControlFlow.CONTINUE:
                    if gid_in not in inners:
                        inners.add(gid_in)
                        stack.append(gid_in)
                elif flow == ControlFlow.BREAK:
                    return None
                else:
                    raise AssertionError(f'Unexpected control flow: {flow}')

        return Beam(gid, inners, terminators, beam_edges)

    def drop_beam(self, beam: Beam, exclude_gids: Set[GradId]):
        for gid in beam.inners.union(beam.terminators):
            if gid not in exclude_gids:
                self.nodes.pop(gid, None)

        for gid_out, gid_in in beam.edges:
            if gid_out in self.edges:
                self.edges[gid_out].remove(gid_in)
                if len(self.edges[gid_out]) == 0:
                    self.edges.pop(gid_out)

    def __repr__(self) -> str:
        dump = ''

        dump += 'Nodes:\n'
        for gid in sorted(self.nodes.keys(), key=lambda x: x.id):
            data = self.nodes[gid]
            data_str = "" if data is None else f': {data}'
            dump += f'    {gid}{data_str}\n'

        dump += '\nEdges:\n'
        for gid_out in sorted(self.edges.keys(), key=lambda x: x.id):
            gids_in = sorted(self.edges[gid_out], key=lambda x: x.id)
            gids_in_str = ', '.join([str(gid_in) for gid_in in gids_in])
            dump += f'    {gid_out} <- [{gids_in_str}]\n'

        return dump


class LogRecord(object):
    def __init__(self, module: nn.Module, input: List[torch.Tensor], output: List[torch.Tensor]):
        self.module = module
        self.input = input
        self.output = output

        d = dict(module.named_parameters(recurse=False))
        self.module_params = {tensor_id(v): k for k, v in d.items()}

        self.input_gids = set([grad_id(t.grad_fn) for t in input])
        self.output_gids = set([grad_id(t.grad_fn) for t in output])

    def __repr__(self):
        class_name = str(self.module.__class__).split(".")[-1].split("'")[0]

        params_str = ', '.join([f'{k}: {v}' for k, v in self.module_params.items()])
        input_str = ', '.join([f'{tensor_id(t)} # {grad_id(t.grad_fn)}' for t in self.input])
        output_str = ', '.join([f'{tensor_id(t)} # {grad_id(t.grad_fn)}' for t in self.output])

        return f'{class_name}({params_str}) - {input_str} => {output_str}'


def introspect(model: nn.Module, input_size):
    def var2list(var):
        if isinstance(var, tuple):
            return list(var)
        else:
            return [var]

    anti_gc = set()  # Needed to prevent the variables from being reused by torch backend

    hook_handles = []

    params = dict(model.named_parameters())
    param_map = {tensor_id(v): k for k, v in params.items()}

    network_graph = NetworkGraph()
    log = []

    def traverse_grad(var):
        anti_gc.add(var)
        var_id = grad_id(var)

        if not network_graph.contains(var_id):
            class_name = str(type(var).__name__)

            if hasattr(var, 'variable'):
                u = var.variable
                variable_id = tensor_id(u)
                variable_size = u.size()
                param_name = param_map.get(variable_id)
            else:
                variable_id = None
                variable_size = None
                param_name = None

            network_graph.add_node(var_id, NodeData(variable_id=variable_id,
                                                    variable_size=variable_size,
                                                    param_name=param_name,
                                                    class_name=class_name))

            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        u_id = traverse_grad(u[0])
                        network_graph.add_edge(u_id, var_id)

            if hasattr(var, 'saved_tensors'):
                for u in var.saved_tensors:
                    u_id = traverse_grad(u)
                    network_graph.add_edge(u_id, var_id)

        return var_id

    def forward_hook(module: nn.Module, input, output):
        input = var2list(input)
        output = var2list(output)

        for entry in (input + output):
            anti_gc.add(entry.grad_fn)

        log.append(LogRecord(module=module, input=input, output=output))

    def register_hook(module: nn.Module):
        handle = module.register_forward_hook(forward_hook)
        hook_handles.append(handle)

    def run_model():
        dtype = torch.cuda.FloatTensor
        input_size_tuple = [input_size] if not isinstance(input_size, tuple) else input_size
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size_tuple]

        model.apply(register_hook)
        y = model(*x)

        for t in var2list(y):
            traverse_grad(t.grad_fn)

        for h in hook_handles:
            h.remove()

    def control_flow(record: LogRecord):
        def check(gid: GradId, data: NodeData) -> ControlFlow:
            if gid in record.input_gids:
                return ControlFlow.STOP

            if data.variable_id is not None:
                return ControlFlow.STOP if data.variable_id in record.module_params else ControlFlow.CONTINUE

            return ControlFlow.CONTINUE

        return check

    run_model()

    print(network_graph)
    for r in log:
        print(r)

    r = log[0]
    beam = network_graph.beam_search(list(r.output_gids)[0], control_flow(r))
    network_graph.drop_beam(beam, r.input_gids)
    print('')
    print(beam)
    print('')
    print(network_graph)

    r = log[1]
    beam = network_graph.beam_search(list(r.output_gids)[0], control_flow(r))
    network_graph.drop_beam(beam, r.input_gids)
    print('')
    print(beam)
    print('')
    print(network_graph)







