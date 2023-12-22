from typing import Tuple, Dict, List, Any, Union
import pandas as pd
import numpy as np
from utils.datastructs import OptDataCEP


def check_column(df, names_array): # Ok used in load_data.py
    for name in names_array:
        # Check existence of necessary column
        if name not in df.columns:
            raise ValueError(f"No column called '{name}' in {repr(df)}")


def get_cep_design_variables(opt_result):
    design_variables = get_cep_variables(opt_result, "dv")
    return design_variables


"""
    get_cep_variables(opt_result::OptResult, variable_type::String)
The variables can be of different type. The different groups of variables can be extracted from the `OptResult` based on the `variable_type`:
Returns all variables which types match the String `variable_type`
"""

def get_cep_variables(opt_result, variable_type: str) -> Dict[str, Any]:
    variables = {}
    for key, val in opt_result.variables.items():
        if val.type == variable_type:
            variables[key] = val
    if not variables:
        raise ValueError(f"{variable_type}-Variable not provided in {opt_result.descriptor}")
    else:
        return variables
    

"""
    set_config_cep(opt_data::OptDataCEP; kwargs...)
kwargs can be whatever you need to run the run_opt
it can hold
  -  `fixed_design_variables`: Dictionary{String,Any}
  -  `transmission`: true or false
  -  `generation`: true or false
  -  `storage_p`: true or false
  -  `storage_e`: true or false
  -  `existing_infrastructure`: true or false
  -  `descritor`: a String like "kmeans-10-co2-500" to describe this CEP-Model
  -  `first_stage_vars`: a Dictionary containing the OptVariables from a previous run
The function also checks if the provided data matches your kwargs options (e.g. it let's you know if you asked for transmission, but you have no tech with it in your data)
Returning Dictionary with the variables as entries
"""
def set_config_cep(opt_data, **kwargs): # OK 
    # Create new dictionary and set possible unique tech_groups to False to later check wrong settings
    
    tech_data = opt_data.techs
    tech_all = tech_data.keys()
    #print(tech_data)  # Corrected the variable name
    
    tech_groups = set()  # Use a set to store unique tech_groups
    for tech in tech_all:
        all_tech_group = tech_data[tech].tech_group
        tech_groups.update(all_tech_group)
        
    #print(tech_groups)
    
    # Check the existence of the tech_group (like generation or storage - see techs.yml) and write it into dictionary
    config = { "demand": False,   "transmission": False, "storage": False,  "conversion": False, "non_dispatchable_generation": False, "dispatchable_generation": False}
    for each_tech_group in tech_groups:
        #if each_tech_group in config:
        config[each_tech_group] = True
    #print(config)
    # Loop through the kwargs and write them into dictionary
    for kwarg in kwargs.items():
        # Check for false combination
        if kwarg[0] in config.keys():
            if config[kwarg[0]] is False and kwarg[1]:
                raise ValueError("Option " + kwarg[0] + " cannot be selected with input data provided for " + opt_data.region)
        config[kwarg[0]] = kwarg[1]
    #print(config)
    return config

"""
    set_config_cep!(config::Dict{String,Any}; kwargs...)
add or replace items to an existing config:
- `fixed_design_variables`: `Dict{String,OptVariable}``
- `slack_cost`: Number
"""

"""
def set_config_cep(config, **kwargs):
    # Loop through the kwargs and add them to the dictionary
    for key, value in kwargs.items():
        #print(kwargs.items())
        config[str(key)] = value

    # Return dictionary with the information
    return config
"""

"""
    check_opt_data_cep(opt_data::OptDataCEP)
Check the consistency of the data
""" 
def check_opt_data_cep(opt_data):
    if opt_data.lines:
        for tech, line in opt_data.lines.keys():
            #print(tech)
            node= opt_data.lines[tech, line].node_end
            #print(node)
            if node not in opt_data.nodes[node].name:
                raise ValueError(f"Node {node} set as ending node, but not included in nodes-Data")

""" 
def check_opt_data_cep(opt_data):
    # Check lines
    # Only when Data provided
    if opt_data.lines:
        # Check existence of start and end node
        for tech,line in opt_data.lines.keys():
            print(tech)
            for node in opt_data.lines[tech,line].node_end:
                print(node)
                if node not in opt_data.nodes[node]:
                    raise ValueError("Node " + node + " set as ending node, but not included in nodes-Data")
"""
"""
    get_total_demand(cep::OptModelCEP, ts_data::ClustData)
Return the total demand by multiplying demand with deltas and weights for the OptModel CEP
"""

def get_total_demand(cep, ts_data):
    # DATA
    set = cep.set
    # ts: Dict( tech-node ): t x k
    ts = ts_data.data
    # ts_weights: k - weight of each period:
    ts_weights = ts_data.weights
    # ts_deltas: t x k - Δt of each segment x period
    ts_deltas = ts_data.delta_t
    total_demand = 0
    for node in set["nodes"]["all"]:
        for t in set["time_T_period"]["all"]:
            for k in set["time_K"]["all"]:
                total_demand += ts["demand_electricity-" + node][t, k] * ts_deltas[t, k] * ts_weights[k]
    return total_demand



"""
    get_cost_series(opt_data::OptDataCEP,scenario::Scenario)
Return an array for the time series of costs in all the impact dimensions and the set of impacts
"""
def get_cost_series(opt_data, scenario):
    return get_cost_series(opt_data.nodes, opt_data.var_costs, scenario.clust_res, scenario.opt_res.model_set, scenario.opt_res.variables)


"""
    get_met_cap_limit(cep::OptModelCEP, opt_data::OptDataCEP, variables::Dict{String,OptVariable})
Return the technologies that meet the capacity limit
"""
def get_met_cap_limit(cep, opt_data, variables):
    # DATA
    set = cep.set
    nodes = opt_data.nodes
    lines = opt_data.lines

    met_cap_limit = []
    # For all
    if "node" in set["tech"]:
        for tech in set["tech"]["node"]:
            for node in set["nodes"]["all"]:
                # Check if the limit is reached in any capacity at any node
                if sum(variables["CAP"][tech, :, node]) == nodes[tech][node].power_lim:
                    # Add this technology and node to the met_cap_limit Array
                    met_cap_limit.append(tech + "-" + node)
    if "line" in set["tech"]:
        for tech in set["tech"]["line"]:
            for line in set["lines"]["all"]:
                # Check if the limit is reached in any capacity at any line
                if sum(variables["TRANS"][tech, :, line]) == lines[tech][line].power_lim:
                    # Add this technology and line to the met_cap_limit Array
                    met_cap_limit.append(tech + "-" + line)
    # If the array isn't empty throw an error (as limits are only for numerical speedup)
    if met_cap_limit:
        # TODO change to warning
        print(f"Limit is reached for techs {met_cap_limit}")
    return met_cap_limit


"""
  push!(set::Dict{String,Array},key::String,value::Any)

Push a `value` into the Array of `set[key]`. If no Array `set[key]` exists, setup new Array
"""

def push(set_dict, key, value):
    if key in set_dict:
        if value not in set_dict[key]:
            set_dict[key].append(value)
    else:
        set_dict[key] = [value]


"""
    get_limit_dir(limit::Dict{String,Number})
The limit_dir is organized as two dictionaries in each other: limit_dir[impact][carrier]='impact/carrier' The first dictionary has the keys of the impacts, the second level dictionary has the keys of the carriers and value of the limit per carrier
"""

def get_limit_dir(limit):
    limit_dir = {}
    for key in limit.keys():
        impact, carrier = key.split("/")
        if impact not in limit_dir:
            limit_dir[impact] = {}
        limit_dir[impact][str(carrier)] = limit["{}/{}".format(impact, carrier)]
    return limit_dir

"""
    text_limit_emission(limit_emission::Dict{String,Dict{String,Number}})
Return text with the information of the `limit_emission` as a text
"""
def text_limit_emission(limit_emission: dict) -> str:
    text_limit = ""
    for emission, carriers in limit_emission.items():
        for carrier, value in carriers.items():
            text_limit += f"{emission}-Emission ≤ {value} [kg-{emission}-eq. per MWh-{carrier}],"
    return text_limit









