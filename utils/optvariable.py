import numpy as np
from typing import List, Tuple, Dict, Any
from itertools import product
from utils.datastructs import OptVariable
# Change here

def build_lookup(ax):
    d = {}
    cnt = 1
    for el in ax:
        if el in d:
            raise ValueError(f"Repeated index {el}. Index sets must have unique elements.")
        d[el] = cnt
        cnt += 1
    return d


"""
# Description from Julia version
    OptVariable(cep::OptModelCEP, variable::Symbol, type::String)
Constructor for OptVariable taking JuMP Array and type (ov-operational variable or dv-decision variable)
"""

def OptVariable(cep, variable, type, scale: Dict, round_sigdigits=8):
    jumparray = np.array([value for value in cep.model[variable]])
    #Get the scaled optvar from model and turn DenseAxisArray and SparseAxisArray into Dense OptVariable
    scaled_optvar = OptVariable(jumparray, type, cep.set)
    #Unscale the data based on the scaling parameters in Dictionary scaleend
    unscaled_data = scaled_optvar.data * scale[variable]
    #Return Optvariable with the unscaled data
    return OptVariable(np.round(unscaled_data, round_sigdigits), scaled_optvar.axes, axes_names=scaled_optvar.axes_names, type=scaled_optvar.type)



def OptVariable(jumparray, type, set_dict):
        axes_number = len(list(jumparray.data.keys())[0])
        axes_names = []
        for axe_number in range(1, axes_number+1):
            # Get the unique values in each dimension of the indexing tuple
            v = np.unique([key[axe_number-1] for key in jumparray.data.keys()])
            axes_names.append(get_axes_name(set_dict, v))
        # Get the axes for the optVariable
        axs = get_axes(set_dict, axes_names)
        # Create an OptVariable with zeros
        optvar = OptVariable(np.zeros(tuple([len(ax) for ax in axs])), *axs, axes_names=axes_names, type=type)
        # Fill data from spare jumparray into dense array
        for idx in jumparray.keys():
            # Find corresponding index of dense array
            dense_idx = []
            for i in range(len(idx)):
                dense_idx.append(np.where(optvar.axes[i] == idx[i])[0][0])
            # Overwrite the
            optvar.data[tuple(dense_idx)] = jumparray[idx]
        return optvar 


# TODO: Add functions get_axes_name,get_axes  


def OptVariable(jumparray, type, set_dict):
            axes_names = []
            for axe in jumparray.axes:
                for name, val in set_dict.items():
                    for n, v in val.items():
                        if axe == v:
                            axes_names.append(name)
                            break
            return OptVariable(jumparray.data, *jumparray.axes, axes_names=axes_names, type=type)
        

# Added today
"""
    OptVariable(data::Array{T, N}, axes...) where {T, N}

Construct a OptVariable array with the underlying data specified by the `data` array
and the given axes. Exactly `N` axes must be provided, and their lengths must
match `size(data)` in the corresponding dimensions.
"""

def OptVariable(data: np.ndarray, *axs: np.ndarray, axes_names: List[str] = None, type: str = "") -> np.ndarray:
    assert len(axs) == data.ndim
    if axes_names is None:
        axes_names = [""] * data.ndim
    lookup = [build_lookup(ax) for ax in axs]
    return OptVariable(data, axs, lookup, axes_names, type)


"""
    OptVariable{T}(undef, axes...) where T

Construct an uninitialized OptVariable with element-type `T` indexed over the
given axes.
"""
#def OptVariable(T, undef, *axs, axes_names=[""], type=""):
#    return construct_undef_array(T, axs, axes_names=axes_names, type=type)

#def construct_undef_array(T, axs, axes_names, type):
#    return OptVariable(np.empty(tuple([len(ax) for ax in axs]), dtype=T), *axs, axes_names=axes_names, type=type)

def OptVariable(T, undef, *axs, axes_names=[""], type=""):
    return construct_undef_array(T, axs, axes_names=axes_names, type=type)

def construct_undef_array(T, axs, axes_names, type):
    return OptVariable(np.empty(tuple([len(ax) for ax in axs]), dtype=T), *axs, axes_names=axes_names, type=type)

def isempty(A):
    return len(A.data) == 0

def size(A):
    return A.data.shape

def LinearIndices(A):
    raise Exception("OptVariable does not support this operation.")

def axes(A):
    return A.axes

def axes(A, dims):
    return A.axes[A.axes_names.index(dims)]

def CartesianIndices(a):
    return np.ndindex(a.shape)

# Indexing

def isassigned(A, *idx):
    return len(idx) == len(A.shape) and all([t[1] in A.lookup and t[2] in A.lookup[t[1]] for t in enumerate(idx)])

def eachindex(A):
    return np.ndindex(A.shape)

def lookup_index(i, lookup):
    if isinstance(i, slice):
        return slice(None, None, None)
    else:
        return lookup[i]

def _to_index_tuple(idx, lookup):
    if len(idx) == 0:
        return ()
    else:
        return (lookup_index(idx[0], lookup), *_to_index_tuple(idx[1:], lookup[1:]))

def to_index(A, *idx):
    return _to_index_tuple(idx, [A.lookup[i] for i in range(len(idx))])

def has_colon(idx):
    if len(idx) == 0:
        return False
    else:
        return isinstance(idx[0], slice) or has_colon(idx[1:])

def getindex(A, *idx):
    if has_colon(idx):
        return A.data[to_index(A, *idx)]
    else:
        return A.data[to_index(A, *idx)]

def setindex(A, v, *idx):
    A.data[to_index(A, *idx)] = v

def IndexStyle(T):
    return "IndexAnyCartesian"



"""
    OptVariableKey

Structure to hold a OptVariable key when it is viewed as key-value collection.
"""

class OptVariableKey:
        def __init__(self, I: Tuple):
            self.I = I

        def __getitem__(self, args):
            return self.I[args]

class OptVariableKeys:
        def __init__(self, product_iter):
            self.product_iter = product_iter

        def __len__(self):
            return len(self.product_iter)

        def __iter__(self):
            for item in self.product_iter:
                yield OptVariableKey(item)

def keys(a):
        return OptVariableKeys(product(*a.axes))

def getindex(a, k):
        return a[tuple(k.I)]


"""
    get_axes_name(set::Dict{String,Any},values::Array)

Figure out the set within the dictionary `set`, which has equivalent elements to the provided `values`.
The `set` has to be organized as follows: Each entry `set[set_name]` can either be:
- a set-element itself, which is an Array or UnitRange
- or a dictionary with set-subgroups for this set. The set-subgroup has to have a set element called `set[set_name]["all"]`, which contains an Array or UnitRange containing all values for the set_name
"""

def get_axes_name(set_dict, values):
            for k, v in set_dict.items():
                # Check the group 'all' first
                if get_axes_name(v['all'], values):
                    return k
                # Check the subsets second
                for kk, vv in v.items():
                    if get_axes_name(vv, values):
                        return k
            raise ValueError(f"The values {values} were not found in set {set_dict}")

def get_axes_name(set_element, values):
            if sorted(set_element) == sorted(values):
                return True

def get_axes_name(set_element, values):
            if list(set_element) == sorted(values):
                return True


"""
    get_axes(set::Dict{String,Dict{String,Array}}, axes_names::Array{String,1})


get the axes defined by the `axes_names` from the `set`
"""
def get_axes(set_dict, axes_names):
    axes = []
    for axes_name in axes_names:
        axes.append(set_dict[axes_name]["all"])
    return axes




