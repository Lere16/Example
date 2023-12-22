from typing import Dict
from utils.datastructs import OptDataCEP, OptVariableLine

def get_opt_set_names(config: dict):
    # Define all set-names that are always included
    set_names = ["nodes", "carrier", "tech", "impact", "year", "account", "infrastruct", "time_K", "time_T_point", "time_T_period"]
    # Add set-names that are specific for certain configurations
    if config["transmission"]:
        set_names.append("lines")
        set_names.append("dir_transmission")
    if config["seasonalstorage"]:
        set_names.append("time_I_point")
        set_names.append("time_I_period")
    return set_names



def setup_opt_set(ts_data, opt_data, config):
    """
    Fetch sets from the time series (ts_data) and capacity expansion model data (opt_data) and return a dictionary `set`.

    The dictionary is organized as:
    - `set[tech_name][tech_group]=[elements...]`
    - `tech_name` is the name of the dimension like e.g. `tech`, or `node`
    - `tech_group` is the name of a group of elements within each dimension like e.g. `["all", "generation"]`. The group `'all'` always contains all elements of the dimension
    - `[elements...]` is the Array with the different elements like `["pv", "wind", "gas"]`
    """
    # costs::OptVariable: costs[tech, node, year, account, impact] - annulized costs [USD in USD/MW_el, CO2 in kg-CO₂-eq./MW_el]
    costs = opt_data.costs
    # techs::OptVariable: techs[tech] - OptDataCEPTech
    techs = opt_data.techs
    # nodes::OptVariable: nodes[tech, node] - OptDataCEPNode
    nodes = opt_data.nodes
    # lines::OptVariable: lines[tech, line] - OptDataCEPLine
    lines = opt_data.lines
    lines=OptVariableLine(lines)

    # Create dictionaries for each set_name
    sets = {}
    for set_name in get_opt_set_names(config):
        sets[set_name] = {}
    #print(get_opt_set_names(config))
    # tech - dimension of elements that can generate, transmit, store, or demand
    setup_opt_set_tech(sets, opt_data, config)
    # carrier - dimension of energy-carriers like `electricity`, `hydrogen`
    setup_opt_set_carrier(sets, opt_data, config)
    # impact - dimension of impact categories (first is monetary)
    setup_opt_set_impact(sets,opt_data,config)
    sets["nodes"]["all"] = list(nodes.keys())
    # year - annual time dimension
    sets["year"]["all"] = list(costs.year)
    # account - dimension of cost calculation: capacity&fixed costs that do not vary with the generated power or variable costs that do vary with generated power
    sets["account"]["all"] = list(costs.account)
    
    if config["transmission"]:
        # lines - dimension of spacial resolution for elements along lines
        sets["lines"]["all"]=list(lines.line)
        #dir_transmission - dimension of directions along a transmission line
        sets["dir_transmission"]["all"]=["uniform","opposite"]
    # infrastruct - dimension of status of infrastructure new or existing
    sets["infrastruct"]["all"] = ["new", "ex"]
    # time_K - dimension of clustered time-series periods
    sets["time_K"]["all"] = list(range(1, ts_data.K+1))
    # time_T - dimension of time within each clustered time-series period:
    sets["time_T_period"]["all"] = list(range(1, ts_data.T+1))
    # or the number of the points in time    <0>---<1>---<2>...
    sets["time_T_point"]["all"] = list(range(ts_data.T+1))
    
    #TODO: If seasonal storage is implemented, add the following sets:
    #print(sets)
    return sets


def setup_opt_set_tech(sets, opt_data, config):
    # `techs::OptVariable`: techs[tech] - OptDataCEPTech
    techs = opt_data.techs

    sets["tech"]["limit"] = []
    # Loop through all techs
    for tech in techs:
        # Check if the primary tech-group (first entry within the Array of tech-groups) is in config
        if config[techs[tech].tech_group[0]]:
            if "all" not in sets["tech"]:
                sets["tech"]["all"]=[]
            # Add the technology to the set of all techs
            sets["tech"]["all"].append(tech)
            # Loop through all tech_groups for this technology
            for tech_group_name in techs[tech].tech_group:
                # Initialize the lists if they don't exist
                if tech_group_name not in sets["tech"]:
                    sets["tech"][tech_group_name] = []
                sets["tech"][tech_group_name].append(tech)
                    
                if "existing" in config["infrastructure"]:
                    if "exist_inf" not in sets["tech"]:
                        sets["tech"]["exist_inf"] = []
                    sets["tech"]["exist_inf"].append(tech)
                    if "no_exist_inf" not in sets["tech"]:
                        sets["tech"]["no_exist_inf"] = []
                    sets["tech"]["no_exist_inf"].append(tech)

                # Add this tech to the tech-group "limit", if the tech_group is a key within the dictionary `techgroups_limit_inf`
                if "limit" in config["infrastructure"]:
                    if tech_group_name in config["infrastructure"]["limit"]:
                        sets["tech"]["limit"].append(tech)
    #print(sets)
            # Push this tech to the set: `set[tech_unit]` - the unit describes if the capacity of this tech is either set up as power or energy capacity
            if techs[tech].unit not in sets["tech"]:
                sets["tech"][techs[tech].unit] = []
            sets["tech"][techs[tech].unit].append(tech)
            #print(techs[tech].unit)
    #print(sets)
            # Push this tech to the set: `set[tech_structure]` - the structure describes if the capacity of this tech is either set up on a node or along a line
            if techs[tech].structure not in sets["tech"]:
                sets["tech"][techs[tech].structure] = []
            sets["tech"][techs[tech].structure].append(tech)

            # Push the technology as an input-carrier
            if "carrier" in techs[tech].input:
                if techs[tech].input["carrier"] not in sets["tech"]:
                    sets["tech"][techs[tech].input["carrier"]] = []
                # Add this technology to the group of the carrier
                sets["tech"][techs[tech].input["carrier"]].append(tech)
                sets["tech"][techs[tech].input["carrier"]]=list(set(sets["tech"][techs[tech].input["carrier"]]))
    #print(sets)
            # Push the technology as an output-carrier
            if "carrier" in techs[tech].output:
                if techs[tech].output["carrier"] not in sets["tech"]:
                    sets["tech"][techs[tech].output["carrier"]] = []
                # Add this technology to the group of the carrier
                sets["tech"][techs[tech].output["carrier"]].append(tech)
                sets["tech"][techs[tech].output["carrier"]]=list(set(sets["tech"][techs[tech].output["carrier"]]))
                
    #print(sets)
    # Test if demand exists
    if not sets["tech"]["demand"] and sets["tech"]["exist_inf"]:
        print("No existing demand is provided - Ensure that `run_opt` option `infrastructure` is correct and looks like:  `Dict{String,Array}(\"existing\"=>[\"demand\", ...])` or `Dict{String,Array}(\"existing\"=>[\"all\"])`")
    sets["tech"]["all"]=list(set(sets["tech"]["all"]))
    sets["tech"]["exist_inf"]=list(set(sets["tech"]["exist_inf"]))
    sets["tech"]["no_exist_inf"]=list(set(sets["tech"]["no_exist_inf"]))
    #print(sets)
    return sets


"""
    setup_opt_set_carrier!(ts_data::ClustData,opt_data::CEPData,config::Dict{String,Any})
Add the entry set["carrier"]
"""
def setup_opt_set_carrier(sets, opt_data, config):
    techs = opt_data.techs
    #sets["carrier"] = []
    #sets["carrier"] = {"all": [], "lost_load": []}
    for tech in sets["tech"]["all"]:
        if "carrier" in techs[tech].input:
            #sets["carrier"].append((tech, techs[tech].input["carrier"]))
            if tech not in sets["carrier"]:
                sets["carrier"][tech]=[]
            sets["carrier"][tech].append(techs[tech].input["carrier"])
        
        if "carrier" in techs[tech].output:
            #sets["carrier"].append((tech, techs[tech].output["carrier"]))
            if tech not in sets["carrier"]:
                sets["carrier"][tech]=[]
            sets["carrier"][tech].append(techs[tech].output["carrier"])
        sets["carrier"][tech]=list(set(sets["carrier"][tech]))
    #print(sets)
    if "all" not in sets["carrier"]:
        sets["carrier"]["all"] = []
    sets["carrier"]["all"] = list(set().union(*sets["carrier"].values()))
    #print(sets)
    if "lost_load" not in sets["carrier"]:
        sets["carrier"]["lost_load"]=[]
    sets["carrier"]["lost_load"] = list(set(carrier for carrier in sets["carrier"]["all"] if carrier in config["lost_load_cost"]))
    #sets["carrier"]["lost_load"] = sets["carrier"]["all"].intersection(config["lost_load_cost"].keys())
    #print(sets)
    for k, v in sets["tech"].items():
        for tech in v:
            for carrier in sets["carrier"][tech]:
                if k not in sets["carrier"]:
                    sets["carrier"][k]=[]
                sets["carrier"][k] = list(sets["carrier"][k])
                sets["carrier"][k].append(carrier)
                sets["carrier"][k]=list(set(sets["carrier"][k]))
    #print(sets)
    return sets


"""
    setup_cep_opt_set_impact!(ts_data::ClustData,opt_data::CEPData,config::Dict{String,Any})
Add the entry set["impact"]
"""

def setup_opt_set_impact(sets, opt_data, config):
    # `costs::OptVariable`: costs[tech,node,year,account,impact] - annulized costs [USD in USD/MW_el, CO2 in kg-CO₂-eq./MW_el]`
    costs = opt_data.costs
    #print(costs)
    ## IMPACT ##
    # Ensure that group `limit` and `lost_emission` exist empty
    sets["impact"]["limit"] = []
    sets["impact"]["lost_emission"] = []
    # Loop through all impacts
    for impact in costs.impact:
        if "all" not in sets["impact"]:
            sets["impact"]["all"] = []
        # Add impacts to the set of all impacts
        sets["impact"]["all"].append(impact)
        #print(sets)
        # The first impact shall always be monetary impact
        if impact == costs.impact[0]:
            if "mon" not in sets["impact"]:
                sets["impact"]["mon"] = []
            sets["impact"]["mon"].append(impact)
        # All other impacts are environmental impacts
        else:
            if "env" not in sets["impact"]:
                sets["impact"]["env"] = []
            sets["impact"]["env"].append(impact)
        # Add impact to tech-group `limit` if impact is a key within the `limit_emission`
        if impact in config["limit_emission"]:
            sets["impact"]["limit"].append(impact)
        # Add impact to tech-group `lost_emission` if impact is a key within the `lost_emission_cost`
        if impact in config["lost_emission_cost"]:
            sets["impact"]["lost_emission"].append(impact)
    
    return sets



