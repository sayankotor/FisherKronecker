import torch
import logging
import re
from functools import wraps
from re import Pattern
from typing import Callable, Dict, Optional, Tuple, Sequence

import numpy as np

class SVD(torch.nn.Module):
    def __init__(self,input_size, output_size, r, bias):
        super(SVD, self).__init__()
        self.weight = torch.Tensor()
        self.lin0 = torch.nn.Linear(in_features=input_size, out_features=r, bias=False)
        self.lin1 = torch.nn.Linear(in_features=r, out_features=output_size, bias=bias)

    def forward(self, x):
        output = self.lin0(x)
        output = self.lin1(output)
        return output

def w_svd(module, rank = 51, weight = None):
    bias = module.bias if module.bias is not None else None
    module = module.to('cuda:0',# torch.float32
                       )
    if weight is not None:
        I = torch.diag(torch.sqrt(weight.sum(0))).to(module.weight.device, module.weight.dtype)
    else:
        I = torch.eye(module.in_features).to(module.weight.device, module.weight.dtype)
    
    #U, S, Vt = torch.linalg.svd((I @ module.weight.T).T , full_matrices=False)
    #TODO: speedup fix
    #print('=====\n',module.weight.device,'\n======')
    U, S, Vt = torch.linalg.svd((I @ module.weight.T).T.to(module.weight.device) , full_matrices=False) # driver='gesvdj'

    w1 = torch.linalg.lstsq(I, torch.mm(torch.diag(torch.sqrt(S[0:rank])),Vt[0:rank, :]).T).solution.T
    w2 = torch.mm(U[:, 0:rank], torch.diag(torch.sqrt(S[0:rank])))

    module = SVD(module.in_features,
                    module.out_features,
                    rank, bias is not None)

    module.lin0.weight.data.copy_(w1)
    module.lin1.weight.data.copy_(w2)
    
    if bias is not None:
        module.lin1.bias.data.copy_(bias)
    
    return module

def old_w_svd(module, rank = 51, weight = None):
    bias = module.bias if module.bias is not None else None
    module = module.to('cuda:0',# torch.float32
                       )
    
    if weight is not None:
        I_row = torch.diag(torch.sqrt(weight.sum(0))).to(module.weight.device, module.weight.dtype)
        I_col = torch.diag(torch.sqrt(weight.sum(1))).to(module.weight.device, module.weight.dtype)
        #I_row = torch.diag(torch.prod(weight, 0)).to(module.weight.device, module.weight.dtype)
        #I_col = torch.diag(torch.prod(weight, 1)).to(module.weight.device, module.weight.dtype)
    else:
        I_row = torch.eye(module.in_features).to(module.weight.device, module.weight.dtype)
        I_col = torch.eye(module.out_features).to(module.weight.device, module.weight.dtype)


    #logging.info(str(I_row.shape) + str(module.weight.shape) + str(I_col.shape))

    U, S, Vt = torch.linalg.svd( (I_col @ module.weight @ I_row ).to(module.weight.device) , full_matrices=False, driver='gesvdj')

    w1 = torch.linalg.lstsq(I_row, torch.mm(torch.diag(torch.sqrt(S[0:rank])),Vt[0:rank, :]).T).solution.T
    w2 = torch.linalg.lstsq(I_col, torch.mm(U[:, 0:rank], torch.diag(torch.sqrt(S[0:rank]))) ).solution

    #w1 = torch.mm(torch.diag(torch.sqrt(S[0:rank])),Vt[0:rank, :])
    #w2 = torch.mm(U[:, 0:rank], torch.diag(torch.sqrt(S[0:rank])))

    #w2 = torch.mm(U[:, 0:rank], torch.diag(torch.sqrt(S[0:rank]))) @ I_col.pinverse()

    logging.info('ALTERNATIVE FWSVD')

    module = SVD(module.in_features,
                    module.out_features,
                    rank, bias is not None)

    module.lin0.weight.data.copy_(w1)
    module.lin1.weight.data.copy_(w2)
    
    if bias is not None:
        module.lin1.bias.data.copy_(bias)
    
    return module

# def make_contraction(shape, rank, batch_size=32,
#                      seqlen=512) -> ContractExpression:
#     ndim = len(rank) - 1
#     row_shape, col_shape = shape

#     # Generate all contraction indexes.
#     row_ix, col_ix = np.arange(2 * ndim).reshape(2, ndim)
#     rank_ix = 2 * ndim + np.arange(ndim + 1)
#     batch_ix = 4 * ndim  # Zero-based index.

#     # Order indexes of cores.
#     cores_ix = np.column_stack([rank_ix[:-1], row_ix, col_ix, rank_ix[1:]])
#     cores_shape = zip(rank[:-1], row_shape, col_shape, rank[1:])

#     # Order indexes of input (contraction by columns: X G_1 G_2 ... G_d).
#     input_ix = np.insert(row_ix, 0, batch_ix)
#     input_shape = (batch_size * seqlen, ) + row_shape

#     # Order indexes of output (append rank indexes as well).
#     output_ix = np.insert(col_ix, 0, batch_ix)
#     output_ix = np.append(output_ix, (rank_ix[0], rank_ix[-1]))

#     # Prepare contraction operands.
#     ops = [input_shape, input_ix]
#     for core_ix, core_shape in zip(cores_ix, cores_shape):
#         ops.append(core_shape)
#         ops.append(core_ix)
#     ops.append(output_ix)
#     ops = [tuple(op) for op in ops]

#     return contract_expression(*ops)

# class TTCompressedLinear(torch.nn.Module):
#     """Class TTCompressedLinear is a layer which represents a weight matrix of
#     linear layer in factorized view as tensor train matrix.

#     >>> linear_layer = T.nn.Linear(6, 6)
#     >>> tt_layer = TTCompressedLinear \
#     ...     .from_linear(linear_layer, rank=2, shape=((2, 3), (3, 2)))
#     """

#     def __init__(self, cores: Sequence[torch.Tensor],
#                  bias: Optional[torch.Tensor] = None):
#         super().__init__()

#         for i, core in enumerate(cores):
#             if core.ndim != 4:
#                 raise ValueError('Expected number of dimensions of the '
#                                  f'{i}-th core is 4 but given {cores.ndim}.')

#         # Prepare contaction expression.
#         self.rank = (1, ) + tuple(core.shape[3] for core in cores)
#         self.shape = (tuple(core.shape[1] for core in cores),
#                       tuple(core.shape[2] for core in cores))
#         self.contact = make_contraction(self.shape, self.rank)

#         # TT-matrix is applied on the left. So, this defines number of input
#         # and output features.
#         self.in_features = np.prod(self.shape[0])
#         self.out_features = np.prod(self.shape[1])

#         # Create trainable variables.
#         self.cores = torch.nn.ParameterList(torch.nn.Parameter(core) for core in cores)
#         self.bias = None
#         if bias is not None:
#             if bias.size() != self.out_features:
#                 raise ValueError(f'Expected bias size is {self.out_features} '
#                                  f'but its shape is {bias.shape}.')
#             self.bias = torch.nn.Parameter(bias)

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         # We need replace the feature dimension with multi-dimension to contact
#         # with TT-matrix.
#         input_shape = input.shape
#         input = input.reshape(-1, *self.shape[0])

#         # Contract input with weights and replace back multi-dimension with
#         # feature dimension.
#         output = self.contact(input, *self.cores)
#         output = output.reshape(*input_shape[:-1], self.out_features)

#         if self.bias is not None:
#             output += self.bias
#         return output

#     @classmethod
#     def from_linear(cls, linear: torch.nn.Linear,
#                     shape: Tuple[Tuple[int], Tuple[int]], rank: int, **kwargs):
#         ndim = len(shape[0])

#         # Prepare information about shape and rank of TT (not TTM).
#         tt_rank = (1, ) + (rank, ) * (ndim - 1) + (1, )
#         tt_shape = tuple(n * m for n, m in zip(*shape))

#         # Reshape weight matrix to tensor indexes like TT-matrix.
#         matrix = linear.weight.data.T
#         tensor = matrix.reshape(shape[0] + shape[1])
#         for i in range(ndim - 1):
#             tensor = tensor.moveaxis(ndim + i, 2 * i + 1)

#         # Reshape TT-matrix to a plain TT and apply decomposition.
#         tensor = tensor.reshape(tt_shape)
#         cores = ttd(tensor, tt_rank, **kwargs)

#         # Reshape TT-cores back to TT-matrix cores (TTM-cores).
#         core_shapes = zip(tt_rank, *shape, tt_rank[1:])
#         cores = [core.reshape(core_shape)
#                  for core, core_shape in zip(cores, core_shapes)]

#         # Make copy of bias if it exists.
#         bias = None
#         if linear.bias is not None:
#             bias = torch.clone(linear.bias.data)

#         return TTCompressedLinear(cores, bias)

#     @classmethod
#     def from_random(cls, shape: Tuple[Tuple[int], Tuple[int]], rank: int,
#                     bias: bool = True):
#         tt_ndim = len(shape[0])
#         tt_rank = (1, ) + (rank, ) * (tt_ndim - 1) + (1, )
#         core_shapes = zip(tt_rank, *shape, tt_rank[1:])
#         cores = [torch.randn(core_shape) for core_shape in core_shapes]

#         bias_term = None
#         if bias:
#             out_features = np.prod(shape[1])
#             bias_term = torch.randn(out_features)

#         return TTCompressedLinear(cores, bias_term)

def map_module(root: torch.nn.Module,
               func: Callable[[torch.nn.Module, str], torch.nn.Module],
               patt: Optional[str] = None) -> torch.nn.Module:
    """Function ``map_module`` applies a function to each leaf of module tree
    which matches to a specified pattern.

    Parameters
    ----------
    root : torch.nn.Module
        Module to modify.
    func : callable
        Function to be applied to every module (or matched to pattern) in
        module tree.
    patt : str, optional
        Pattern to filter modules by path in module tree.

    Returns
    -------
    torch.nn.Module
        Module modified in-place.
    """
    @wraps(func)
    def func_safe(*args, **kwargs):
        node = func(*args, **kwargs)
        if not isinstance(node, torch.nn.Module):
            raise ValueError('Mapped result must be toch.nn.Module type '
                             f'but given {type(node)}.')
        return node

    return _map_module(root, func_safe, re.compile(patt or r'.*'), '')


def _map_module(root: torch.nn.Module,
                func: Callable[[torch.nn.Module, str], torch.nn.Module], patt: Pattern,
                path: str) -> torch.nn.Module:
    #logging.info('Try to apply compression to layer %s', path)
    for name, child in root.named_children():
        node = _map_module(child, func, patt, f'{path}/{name}')
        if node != child:
            setattr(root, name, node)
    if patt.match(path or '/'):
        root = func(root, path or '/')
    return root


def convert_linear(module: torch.nn.Linear, ctor, **kwargs) -> torch.nn.Module:
    """Function convert_linear takes module and returns linear module with
    approximate matmul. Non-linear modules are returned intact.
    """
    if not isinstance(module, torch.nn.Linear):
        return module
    raise NotImplementedError


def numel(module: torch.nn.Module):
    value = sum(x.numel() for x in module.parameters()) + \
            sum(x.numel() for x in module.buffers())

    def account_prunned(module: torch.nn.Module, path: str):
        nonlocal value
        for name, attr in vars(module).items():
            if not name.endswith('_mask') or not isinstance(attr, torch.Tensor):
                continue

            weight_name = name[:-5]
            if not hasattr(module, weight_name):
                continue

            weight = getattr(module, weight_name)
            value -= weight.numel() - attr.sum()
            value += attr.numel()
        return module

    def account_quantized(module: torch.nn.Module, path: str):
        nonlocal value
        if isinstance(module, torch.nn.quantized.Linear):
            value += module.weight().numel()
            if module.bias() is not None:
                value += module.bias().numel()
        return module

    def account_rest(module: torch.nn.Module, path: str):
        account_prunned(module, path)
        account_quantized(module, path)
        return module

    map_module(module, account_rest)
    return value


def sizeof(module: torch.nn.Module):
    value = sum(x.numel() * x.element_size() for x in module.parameters()) + \
            sum(x.numel() * x.element_size() for x in module.buffers())

    def account_prunned(module: torch.nn.Module, path: str):
        nonlocal value
        for name, attr in vars(module).items():
            if not name.endswith('_mask') or not isinstance(attr, torch.Tensor):
                continue

            weight_name = name[:-5]
            if not hasattr(module, weight_name):
                continue

            weight = getattr(module, weight_name)
            value -= (weight.numel() - attr.sum()) * weight.element_size()
            value += attr.numel() * attr.element_size()
        return module

    def account_quantized(module: torch.nn.Module, path: str):
        nonlocal value
        if isinstance(module, torch.nn.quantized.Linear):
            value += module.weight().numel() * module.weight().element_size()
            if (bias := module.bias()) is not None:
                value += bias.numel() * bias.element_size()
        return module

    def account_rest(module: torch.nn.Module, path: str):
        account_prunned(module, path)
        account_quantized(module, path)
        return module

    map_module(module, account_rest)
    return value


def flatten_module(module: torch.nn.Module, regexp=None) -> Dict[str, torch.nn.Module]:
    modules = {}
    map_module(module, lambda x, y: modules.update(**{y: x}) or x, regexp)
    return modules


def print_flatten(module: torch.nn.Module):
    paths = []
    path_len = 0
    names = []
    name_len = 0
    indx_len = 0

    def func(module, path):
        nonlocal path_len, name_len, indx_len
        paths.append(path)
        path_len = max(path_len, len(path))
        name = module.__class__.__name__
        names.append(name)
        name_len = max(name_len, len(name))
        indx_len += 1
        return module

    map_module(module, func)

    indx_len = int(np.ceil(np.log10(indx_len)))
    fmt = f'{{indx:>{indx_len}s}} {{path:{path_len}s}} {{name:{name_len}s}}'
    print(fmt.format(indx='#', path='Path', name='Layer'))
    print('-' * (indx_len + path_len + name_len + 2))
    for i, (path, name) in enumerate(zip(paths, names)):
        print(fmt.format(indx=str(i), path=path, name=name))


def compress_linear(module: torch.nn.Module, path: str,
                       shape: Tuple[Tuple[int], Tuple[int]],
                       rank: int,
                       weight,
                       random_init,
                       cholesky: bool = False
                       ) -> torch.nn.Module:
    if not isinstance(module, torch.nn.Linear):
        return module    
    logging.info('apply svd compression to layer %s', path)

    if random_init:
        module.weight.data.uniform_(0.0, 1.0)
        
    if weight:
        path_fisher = path.replace(r'/','.')[1:]
        logging.info("with weight! for layer %s", path_fisher)
        assert path_fisher in weight, f'{path_fisher} not in weight'
        #with open(weight, 'rb') as f:
        #    mass = pickle.load(f)
        # path_fisher = path.replace(r'/','.')[1:]
        return w_svd(module, rank, weight[path_fisher])
    else:
        return w_svd(module, rank)