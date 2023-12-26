
from utils.utilsfun import get_limit_dir, set_config_cep, check_column
#from gurobipy import Model, GRB
from opt_problems.set import setup_opt_set
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum
import pandas as pd
import numpy as np
import os
from utils.datastructs import OptDataCEPNode, LatLon



def run_opt(ts_data, opt_data, optimizer, descriptor="", storage_type="none", demand=True, dispatchable_generation=True, non_dispatchable_generation=True, conversion=False, transmission=False, lost_load_cost={}, lost_emission_cost={}, limit_emission={}, infrastructure={"existing": ["all"], "limit": []}, scale={":COST": 1e9, ":CAP": 1e3, ":GEN": 1e3, ":SLACK": 1e3, ":INTRASTOR": 1e3, ":INTERSTOR": 1e6, ":FLOW": 1e3, ":TRANS": 1e3, ":LL": 1e6, ":LE": 1e9}, print_flag=False, optimizer_config={}, round_sigdigits=9, time_series_config={}):
    # Activated seasonal or simple storage corresponds with storage
    if storage_type == "simple":
        storage = True
        seasonalstorage = False
    elif storage_type == "none":
        storage = False
        seasonalstorage = False
    else:
        storage = False
        seasonalstorage = False
        print("String indicating `storage_type` not identified")

    limit_emission = get_limit_dir(limit_emission)

    # Setup the config file based on the data input and
    config = set_config_cep(opt_data, descriptor=descriptor, limit_emission=limit_emission, lost_load_cost=lost_load_cost, lost_emission_cost=lost_emission_cost, infrastructure=infrastructure, demand=demand, non_dispatchable_generation=non_dispatchable_generation, dispatchable_generation=dispatchable_generation, storage=storage, conversion=conversion, seasonalstorage=seasonalstorage, transmission=transmission, scale=scale, print_flag=print_flag, optimizer_config=optimizer_config, round_sigdigits=round_sigdigits, region=opt_data.region, time_series={"years": ts_data.years, "K": ts_data.K, "T": ts_data.T, "config": time_series_config, "weights": ts_data.weights, "delta_t": ts_data.delta_t})
    # Run the optimization problem
    run_opt_main(ts_data, opt_data, config, optimizer)
    
    

def run_opt_main(ts_data, opt_data, config, optimizer):
    optimizer_config = config["optimizer_config"]
    cep = Container(working_directory=os.path.join(os.getcwd(), "debugg"))
    info = [config["descriptor"]]
    sets=setup_opt_set(ts_data, opt_data, config)

    
    ##-------------------DEFININING SETS-------------------##
    """ Sets are defined in GAMSPy"""
    #defining sets
    account=Set(cep, name = "account", records =sets["account"]["all"], description = "fixed costs for installation and yearly expenses, variable costs")
    impact=Set(cep, name= "impact", records =sets["impact"]["all"], description = "impact categories like EUR or USD, CO2")
    tech=Set(cep, name= "tech", records =sets["tech"]["all"], description = "generation, conversion, storage, and transmission technologies")
    infrastruct=Set(cep, name= "infrastructur", records =sets["infrastruct"]["all"], description = "infrastructure status being either new or existing")
    node= Set(cep, name= "node", records =sets["nodes"]["all"], description = "spacial energy system nodes")
    carrier= Set(cep, name= "carrier", records =sets["carrier"]["all"], description = "carrier that an energy balance is calculated for electricity, electrictity_battery")
    t= Set(cep, name= "t", records =sets["time_T_period"]["all"], description = "numeration of the time intervals within a period: typically hours")
    k=Set(cep, name= "k", records =sets["time_K"]["all"], description = "numeration of the representative periods")
    year= Set(cep, name= "year", records =sets["year"]["all"], description = "year of the costs data")
    #defining subsets
    tech_node=Set(cep, name= "tech_node", domain=[tech], records =sets["tech"]["node"], description = "technology pressent at a specific node")
    
    
    
    ##-------------------DEFININING PARAMETER-------------------##
  
    """ Parameters are defined in Python"""
    techs=opt_data.techs
    scale=config["scale"]
    #---------ts_weight
    ts_weights_df =np.array(pd.Series(data=list(ts_data.weights)))
    ts_weights=Parameter(cep, "ts_weights", domain=[k], records =ts_weights_df)
    #---------ts_deltas
    ts_deltas_df =np.array(pd.DataFrame(data=ts_data.delta_t))
    ts_deltas=Parameter(cep, "ts_deltas", domain=[t,k], records =ts_deltas_df)
    
    sign_generation=Parameter(cep, "sign_generation", records =1)
        
    #-------Costs
    costs_df = pd.DataFrame(list(opt_data.costs.data.keys()), columns=["tech", "node", "year", "account", "impact"])
    costs_df["value"]=list(opt_data.costs.data.values())
    #------Time series data
    
    idf=Set(cep, name= "id", records =list(ts_data.data.keys()), description = "identifier for all times series data")
    id_d=list()
    id_o=list()
    for key in ts_data.data.keys():
        if "demand_electricity" in key:
            id_d.append(key)
        else:
            id_o.append(key)
    idf_d=Set(cep, name= "id_d", domain=[idf], records =id_d, description = "identifier for demand times series data")
    idf_o=Set(cep, name= "id_o", domain=[idf],records =id_o, description = "identifier for power times series data")
    
    df=pd.DataFrame(columns=['key','t','k','value'])
    for ke,v in ts_data.data.items():
        df_values=pd.DataFrame(v)
        df_values["key"]=ke
        df_values["t"]=df_values.index+1
        df_values=df_values.melt(id_vars=["key","t"], var_name="k")
        df_values['k']=df_values['k'] +1
        df=pd.concat([df,df_values], axis=0, ignore_index=True)
    
    ts=Parameter(cep, "ts", domain=[idf,t,k], records =df)
    
    ##-------------------DEFININING VARIABLE-------------------##
    COST = Variable(
        container=cep,
        name="COST",
        domain=[account, impact , tech],
        type="Positive",
        description="Costs",)
    
    CAP = Variable(
        container=cep,
        name="CAP",
        domain=[tech, infrastruct , node],
        type="free",
        description="Capacity",)
    
    GEN = Variable(
        container=cep,
        name="GEN",
        domain=[tech, carrier, t, k,node ],
        type="free",
        description="Generation",)
    
        ##------------------- SETUP EQUATIONS-------------------##
    # Adding existing infrastructure to the model from the node table
    data_path2 = os.path.join(os.getcwd(), "data", config['region'])
    nodes_tech=load_cep_data_nodes_2(data_path2,techs)
    tech_ex=Set(cep, name= "tech_ex", domain=[tech], records =set(sets["tech"]["node"]).intersection(sets["tech"]["exist_inf"]), description = "existing technology")
    ex={}
    for tecx in set(sets["tech"]["node"]).intersection(sets["tech"]["exist_inf"]):
        for n in sets['nodes']['all']:
            ex[tecx, n]=nodes_tech[tecx, n].power_ex
            
    ex_df = pd.DataFrame([(key[0], key[1], value) for key, value in ex.items()], columns=['tech', 'node', 'value'])
    EX=Parameter(cep, name= "EX", domain=[tech, node], records =ex_df, description = "existing capacity of each each technology at each node")

    #e25=Equation(container=cep, name="e25", domain=[tech_ex,"ex",node])
    #e25[tech_ex, "ex", node]=CAP[tech_ex, 'ex', node]==EX[tech_ex,node]/scale[':CAP']
    CAP.lo[tech_ex, "ex", node]==EX[tech_ex,node]/scale[':CAP']
    
    if config["demand"]:
        
        tech_demand =Set(cep, name= "tech_demand", domain=[tech], records =sets["tech"]["demand"], description = "demand technology")
        carrier_demand=Set(cep, name= "carrier_demand", domain=[carrier], records =techs['demand'].input['carrier'], description = "carrier demand technology")
        demand_costs_df = costs_df[costs_df['tech'] == 'demand']
        costs_d= Parameter(cep, name = "costs_d", domain=[tech_demand,node,year,account,impact], records = demand_costs_df)
        e1=Equation(container=cep, name="e1", domain=[impact, tech_demand,carrier_demand,year])
        e1[impact, tech_demand,carrier_demand,year]=COST['var',impact, tech_demand]==sign_generation*Sum([t,k,node], 
                                                                        GEN[tech_demand,carrier_demand , t,k,node ]
                                                                        * ts_weights[k]*ts_deltas[t,k]*costs_d[tech_demand, node, year,'var', impact])*scale[':GEN']/scale[':COST']
        
        #Fixed Costs based on installed capacity
        e2=Equation(container=cep, name="e2", domain=[impact, tech_demand,year], description=" Fixed Costs based on installed capacity")
        e2[impact, tech_demand,year]=COST['cap_fix',impact, tech_demand]==Sum([t,k], 
                                                    ts_weights[k]*ts_deltas[t,k] )/8760* Sum(node,
                                                                                             CAP[tech_demand,'new',node]*costs_d[tech_demand,node, year,'cap_fix',impact])*scale[':CAP']/scale[':COST']
        
        #Fix demand 
        e3=Equation(container=cep, name="e3", domain=[tech_demand, carrier_demand, idf_d, t, k, node], description=" Fix the demands")
        e3[tech_demand, carrier_demand, idf_d, t, k, node]=GEN[tech_demand, carrier_demand,t, k, node]== (-1)*Sum([infrastruct],
                                                                   CAP[tech_demand, infrastruct, node])*ts[idf_d, t,k]*scale[':CAP']/scale[':GEN']
     
    if config["non_dispatchable_generation"]:
        tech_ndisp=Set(cep, name= "tech_ndisp", domain=[tech], records =sets["tech"]["non_dispatchable_generation"], description = "non dispatchable technology technology")
        carrier_ndisp=Set(cep, name= "carrier_ndisp", domain=[carrier],records =[[techs[tec].output["carrier"] for tec in sets["tech"]["non_dispatchable_generation"]]], description = "carrier for non disp technology")
        costs_ndisp_df = costs_df[costs_df['tech'].isin(sets["tech"]["non_dispatchable_generation"])]
        costs_ndisp= Parameter(cep, name = "costs_ndisp", domain=[tech_ndisp,node,year,account,impact], records = costs_ndisp_df)
        
        # Calculate Variable Costs
        e4=Equation(container=cep, name="e4", domain=[impact, tech_ndisp,carrier_ndisp,year], description=" variable cost for non disp technology")
        e4[impact, tech_ndisp,carrier_ndisp,year]=COST['var', impact, tech_ndisp]==sign_generation* Sum((t,k,node), 
                                                                GEN[tech_ndisp,carrier_ndisp , t,k,node ]
                                                                *ts_weights[k]*ts_deltas[t,k]*costs_ndisp[tech_ndisp, node, year, 'var', impact])*scale[':GEN']/scale[':COST']
        
        # Calculate Fixed Costs
        e5=Equation(container=cep, name="e5", domain=[impact, tech_ndisp,year], description=" Fixed Costs for non disp technology")
        e5[impact, tech_ndisp,year]=COST["cap_fix",impact, tech_ndisp]==Sum((t,k), 
                                                    ts_weights[k]*ts_deltas[t,k] )/8760* Sum(node,
                                                                                             CAP[tech_ndisp,"new",node]*costs_ndisp[tech_ndisp,node, year,"cap_fix",impact])*scale[':CAP']/scale[':COST']

        ## GENERATION ELECTRICITY ##
        # Limit the generation of non-dispathable generation to the infrastructing capacity of non-dispachable power plants
        e6=Equation(container=cep, name="e6", domain=[tech_ndisp, carrier_ndisp, t, k, node], description="constraint limiting generation of non-dispatchable to be positive")
        e7=Equation(container=cep, name="e7", domain=[tech_ndisp, carrier_ndisp, t,k, node, idf_o], description="constraint limiting generation of non-dispatchable")
        e6[tech_ndisp, carrier_ndisp, t, k, node]=GEN[tech_ndisp, carrier_ndisp, t, k, node]>=0
        e7[tech_ndisp, carrier_ndisp, t,k, node, idf_o]=GEN[tech_ndisp, carrier_ndisp, t, k, node]<= Sum(infrastruct, CAP[tech_ndisp, infrastruct, node])*ts[idf_o, t,k]*scale[':CAP']/scale[':GEN']
        
    
    if config["dispatchable_generation"]:
        tech_disp=Set(cep, name= "tech_disp", domain=[tech], records =sets["tech"]["dispatchable_generation"], description = "dispatchable generation technology")
        carrier_disp=Set(cep, name= "carrier_disp",domain=[carrier], records ='electricity', description = "carrier for disp technology")
        costs_disp_df = costs_df[costs_df['tech'].isin(sets["tech"]["dispatchable_generation"])]
        costs_disp= Parameter(cep, name = "costs_disp", domain=[tech_disp,node,year,account,impact], records = costs_disp_df)
        # Calculate Variable Costs
        e8=Equation(container=cep, name="e8", domain=[impact, tech_disp,carrier_disp,year], description=" variable cost for disp technology")
        e8[impact, tech_disp,carrier_disp,year]=COST['var', impact, tech_disp]==sign_generation* Sum((t,k,node), 
                                                                GEN[tech_disp,carrier_disp , t,k,node ]
                                                                    *ts_weights[k]*ts_deltas[t,k]*costs_disp[tech_disp, node, year, 'var', impact])*scale[':GEN']/scale[':COST']
        
        # Calculate Fixed Costs
        e9=Equation(container=cep, name="e9", domain=[impact, tech_disp,year], description="Fixed Costs for disp technology")
        e9[impact, tech_disp,year]=COST['cap_fix',impact, tech_disp]==Sum((t,k), 
                                                    ts_weights[k]*ts_deltas[t,k] )/8760* Sum(node,
                                                                                             CAP[tech_disp,'new',node]*costs_disp[tech_disp,node, year,'cap_fix',impact])*scale[':CAP']/scale[':COST']

        ## DISPATCHABLE GENERATION ##
        # Limit the generation of dispathables to the infrastructing capacity of dispachable power plants
        
        e10=Equation(container=cep, name="e10", domain=[tech_disp, carrier_disp, t, k, node], description="constraint limiting generation of dispatchable to be positive")
        e11=Equation(container=cep, name="e11", domain=[tech_disp, carrier_disp, t,k,node], description="constraint limiting generation of dispatchable")
        e10[tech_disp, carrier_disp, t, k, node]= GEN[tech_disp, carrier_disp, t, k, node]>=0
        e11[tech_disp, carrier_disp, t,k,node]=GEN[tech_disp, carrier_disp, t, k, node]<= Sum(infrastruct, CAP[tech_disp, infrastruct, node])*scale[':CAP']/scale[':GEN']
    
    
    ## ENERGY BALANCE CONSTRAINT ##
    e27=Equation(container=cep, name="e27", domain=[carrier,t, k, node], description="Energy balance constraints for carrier electricity")
    e27[carrier,t, k, node]=Sum(tech, GEN[tech, carrier, t, k,node])==0
    
    ## SETUP OBJECTIVE ##
    #totalCost = Variable(
    #    cep, name="totalCost", type="positive", description="costs of generation and expansion"
    #)
    #e29=Equation(container=cep, name="e29", type="regular")
    #e29[...] = totalCost == Sum([account, impact, tech],
    #       COST[account, impact, tech],)*scale[':COST']

    ### SOLVE MODEL##
    Expansion = Model(cep, name="Expansion",equations=cep.getEquations(), problem="LP", sense='MIN', objective=Sum([account, impact, tech],
            COST[account, impact, tech],),)
    Expansion.solve()
    Expansion.num_domain_violations
    
      
            
        
        
    

    
def load_cep_data_nodes_2(data_path, techs):
    tab = pd.read_csv(os.path.join(data_path, "nodes.csv"))
    check_column(tab, ["node", "infrastruct"]) # modified
    nodes = {}
    for tech in techs:
        for node in tab["node"].unique():
            name = node
            power_ex = tab.loc[(tab["node"] == node) & (tab["infrastruct"] == "ex"), tech].iloc[0]
            data = tab.loc[(tab["node"] == node) & (tab["infrastruct"] == "lim"), tech]
            power_lim = float('inf') if data.empty else data.iloc[0]
            region = tab.loc[tab["node"] == node, "region"].iloc[0]
            latlon = LatLon(tab.loc[tab["node"] == node, "lat"].iloc[0], tab.loc[tab["node"] == node, "lon"].iloc[0])
            nodes[tech, node] = OptDataCEPNode(name, power_ex, power_lim, region, latlon)
            #nodes[node] = OptDataCEPNode(name, power_ex, power_lim, region, latlon)
    return nodes

    

    


# %%
