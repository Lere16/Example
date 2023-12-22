
from utils.utilsfun import get_limit_dir, set_config_cep, check_column
from opt_problems.set import setup_opt_set
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Problem
import pandas as pd
import numpy as np
import gamspy.math as gams_math
import os
from utils.datastructs import OptDataCEPNode, LatLon



def run_opt(ts_data, opt_data, optimizer, descriptor="", storage_type="none", demand=True, dispatchable_generation=True, non_dispatchable_generation=True, conversion=False, transmission=False, lost_load_cost={}, lost_emission_cost={}, limit_emission={}, infrastructure={"existing": ["all"], "limit": []}, scale={":COST": 1e9, ":CAP": 1e3, ":GEN": 1e3, ":SLACK": 1e3, ":INTRASTOR": 1e3, ":INTERSTOR": 1e6, ":FLOW": 1e3, ":TRANS": 1e3, ":LL": 1e6, ":LE": 1e9}, print_flag=True, optimizer_config={}, round_sigdigits=9, time_series_config={}):
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

    # The limit_dir is organized as two dictionaries in each other: limit_dir[impact][carrier]='impact/carrier'
    # The first dictionary has the keys of the impacts, the second level dictionary has the keys of the carriers and value of the limit per carrier
    limit_emission = get_limit_dir(limit_emission)

    # Setup the config file based on the data input and
    config = set_config_cep(opt_data, descriptor=descriptor, limit_emission=limit_emission, lost_load_cost=lost_load_cost, lost_emission_cost=lost_emission_cost, infrastructure=infrastructure, demand=demand, non_dispatchable_generation=non_dispatchable_generation, dispatchable_generation=dispatchable_generation, storage=storage, conversion=conversion, seasonalstorage=seasonalstorage, transmission=transmission, scale=scale, print_flag=print_flag, optimizer_config=optimizer_config, round_sigdigits=round_sigdigits, region=opt_data.region, time_series={"years": ts_data.years, "K": ts_data.K, "T": ts_data.T, "config": time_series_config, "weights": ts_data.weights, "delta_t": ts_data.delta_t})
    # Run the optimization problem
    run_opt_main(ts_data, opt_data, config, optimizer)
    
    


def run_opt_main(ts_data, opt_data, config, optimizer):
    # Check the consistency of the data provided
    optimizer_config = config["optimizer_config"]
    cep = Container()
    info = [config["descriptor"]]
    #Setup set
    sets=setup_opt_set(ts_data, opt_data, config)

    
    ##-------------------DEFININING SETS-------------------##
    """ Sets are defined in GAMSPy"""
    #defining sets
    account=Set(cep, name = "account", records =sets["account"]["all"], description = "fixed costs for installation and yearly expenses, variable costs")
    impact=Set(cep, name= "impact", records =sets["impact"]["all"], description = "impact categories like EUR or USD, CO 2 − eq")
    tech=Set(cep, name= "tech", records =sets["tech"]["all"], description = "generation, conversion, storage, and transmission technologies")
    infrastruct=Set(cep, name= "infrastructur", records =sets["infrastruct"]["all"], description = "infrastructure status being either new or existing")
    node= Set(cep, name= "node", records =sets["nodes"]["all"], description = "spacial energy system nodes")
    carrier= Set(cep, name= "carrier", records =sets["carrier"]["all"], description = "carrier that an energy balance is calculated for electricity, electrictity_battery")
    t= Set(cep, name= "t", records =sets["time_T_period"]["all"], description = "numeration of the time intervals within a period: typically hours")
    k=Set(cep, name= "k", records =sets["time_K"]["all"], description = "numeration of the representative periods")
    year= Set(cep, name= "year", records =sets["year"]["all"], description = "year of the costs data")
    #defining subsets
    tech_node=Set(cep, name= "tech_node", records =sets["tech"]["node"], description = "technology pressent at a specific node")
    
    
    
    ##-------------------DEFININING PARAMETER-------------------##
  
    """ Parameters are defined in Python"""
    costs=opt_data.costs
    techs=opt_data.techs
    nodes=opt_data.nodes
    lines=opt_data.lines
    scale=config["scale"]
    #---------ts_weight
    ts_weights_df =np.array(pd.Series(data=list(ts_data.weights)))
    ts_weights=Parameter(cep, "ts_weights", domain=[k], records =ts_weights_df)
    #---------ts_deltas
    ts_deltas_df =np.array(pd.DataFrame(data=ts_data.delta_t))
    ts_deltas=Parameter(cep, "ts_deltas", domain=[t,k], records =ts_deltas_df)
    
    sign_generation=Parameter(cep, "sign_generation", records =1)
        
    #-------Costs
    
    costs_df = pd.DataFrame(list(costs.data.keys()), columns=["tech", "node", "year", "account", "impact"])
    
    costs_df["value"]=list(opt_data.costs.data.values())
    costs= Parameter(cep, name = "costs", domain = [tech,node,year,account,impact], description = "Costs of generation, conversion, storage, transmission costs", records = costs_df)
    
    #------Time series data
    
    idf=Set(cep, name= "id", records =list(ts_data.data.keys()), description = "identifier for all times series data")
    id_d=list()
    id_o=list()
    for key in ts_data.data.keys():
        if "demand_electricity" in key:
            id_d.append(key)
        else:
            id_o.append(key)
    idf_d=Set(cep, name= "id_d", records =id_d, description = "identifier for demand times series data")
    idf_o=Set(cep, name= "id_o", records =id_o, description = "identifier for power times series data")
    
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
        domain=[tech_node, infrastruct , node],
        type="Positive",
        description="Capacity",)
    
    GEN = Variable(
        container=cep,
        name="GEN",
        domain=[tech, carrier, t, k,node ],
        type="free",
        description="Generation",)
    
        ##------------------- SETUP EQUATIONS-------------------##
    if config["demand"]:
        
        tech_demand=Set(cep, name= "tech_demand", records =sets["tech"]["demand"], description = "demand technology")
        carrier_demand=Set(cep, name= "carrier_demand", records =techs['demand'].input['carrier'], description = "demand technology")
        # Variables costs
        e1=Equation(container=cep, name="e1", type="regular", domain=["var",impact, tech_demand])
        e1['var', impact, tech_demand]=COST[impact, tech_demand]==sign_generation*Sum([t,k,node], 
                                                                  GEN[tech_demand,carrier_demand , t,k,node ]
                                                                  *ts_weights[k]*ts_deltas[t,k]*costs[tech_demand, node, year,'var', impact])*scale[':GEN']/scale[':COST']
        
        #Fixed Costs based on installed capacity
        e2=Equation(container=cep, name="e2", description=" Fixed Costs based on installed capacity")#domain=[capfix,impact,tech_demand]
        e2=COST["cap_fix",impact, tech_demand]==Sum((t,k), 
                                                    ts_weights[k]*ts_deltas[t,k] )/8760* Sum(node,
                                                                                             CAP[tech_demand,"new",node]*costs[tech_demand,node, year,"cap_fix",impact])*scale[':CAP']/scale[':COST']
        
        # Fix the demand
        e3=Equation(container=cep, name="e3", description=" Fix the demands")# domain= [tech_deman, node, carrier, t, k]
        e3=GEN[tech_demand, carrier_demand,t, k, node]== (-1)* Sum(infrastruct, 
                                                                   CAP[tech_demand, infrastruct, node])*ts[idf_d, t,k]*scale[':CAP']/scale[':GEN']

    
    if config["non_dispatchable_generation"]:
        tech_ndisp=Set(cep, name= "tech_ndisp", records =sets["tech"]["non_dispatchable_generation"], description = "non dispatchable technology technology")
        carrier_ndisp=Set(cep, name= "carrier_ndisp", records =[[techs[tec].output["carrier"] for tec in sets["tech"]["non_dispatchable_generation"]]], description = "carrier for non disp technology")
        
        # Calculate Variable Costs
        e4=Equation(container=cep, name="e4", description=" variable cost for non disp technology")
        e4=COST["var", impact, tech_ndisp]==sign_generation* Sum((t,k,node), 
                                                                  GEN[tech_ndisp,carrier_ndisp , t,k,node ]
                                                                  *ts_weights[k]*ts_deltas[t,k]*costs[tech_ndisp, node, year, "var", impact])*scale[':GEN']/scale[':COST']
       
        # Calculate Fixed Costs
        e5=Equation(container=cep, name="e5", description=" Fixed Costs for non disp technology")#domain=[capfix,impact,tech_demand]
        e5=COST["cap_fix",impact, tech_ndisp]==Sum((t,k), 
                                                    ts_weights[k]*ts_deltas[t,k] )/8760* Sum(node,
                                                                                             CAP[tech_ndisp,"new",node]*costs[tech_ndisp,node, year,"cap_fix",impact])*scale[':CAP']/scale[':COST']
        ## GENERATION ELECTRICITY ##
        # Limit the generation of non-dispathable generation to the infrastructing capacity of non-dispachable power plants
        e6=Equation(container=cep, name="e6", description="constraint limiting generation of non-dispatchable to be positive")
        e7=Equation(container=cep, name="e7", description="constraint limiting generation of non-dispatchable")
        e6=GEN[tech_ndisp, carrier_ndisp, t, k, node]>=0
        e7=GEN[tech_ndisp, carrier_ndisp, t, k, node]<= Sum(infrastruct, CAP[tech_ndisp, infrastruct, node])*ts[idf_o, t,k]*scale[':CAP']/scale[':GEN']
        
        
    if config["dispatchable_generation"]:
        tech_disp=Set(cep, name= "tech_disp", records =sets["tech"]["dispatchable_generation"], description = "dispatchable generation technology")
        carrier_disp=Set(cep, name= "carrier_disp", records =[[techs[tec].output["carrier"] for tec in sets["tech"]["dispatchable_generation"]]], description = "carrier for disp technology")
        # Calculate Variable Costs
        e8=Equation(container=cep, name="e8", description=" variable cost for disp technology")
        e8=COST["var", impact, tech_disp]==sign_generation* Sum((t,k,node), 
                                                                  GEN[tech_disp,carrier_disp , t,k,node ]
                                                                  *ts_weights[k]*ts_deltas[t,k]*costs[tech_disp, node, year, "var", impact])*scale[':GEN']/scale[':COST']
       
        # Calculate Fixed Costs
        e9=Equation(container=cep, name="e9", description="Fixed Costs for disp technology")#domain=[capfix,impact,tech_demand]
        e9=COST["cap_fix",impact, tech_disp]==Sum((t,k), 
                                                    ts_weights[k]*ts_deltas[t,k] )/8760* Sum(node,
                                                                                             CAP[tech_disp,"new",node]*costs[tech_disp,node, year,"cap_fix",impact])*scale[':CAP']/scale[':COST']
                                                    
        ## DISPATCHABLE GENERATION ##
        # Limit the generation of dispathables to the infrastructing capacity of dispachable power plants
        
        e10=Equation(container=cep, name="e10", description="constraint limiting generation of dispatchable to be positive")
        e11=Equation(container=cep, name="e11", description="constraint limiting generation of dispatchable")
        e10=GEN[tech_disp, carrier_disp, t, k, node]>=0
        e11=GEN[tech_disp, carrier_disp, t, k, node]<= Sum(infrastruct, CAP[tech_disp, infrastruct, node])*scale[':CAP']/scale[':GEN']
    
    if config["conversion"]:
        sign_generation=-1
        tech_conv=Set(cep, name= "tech_conv", records =sets["tech"]["conversion"], description = "conversion technology")
        carrier_conv=Set(cep, name= "carrier_conv", records =[techs[tec].input["carrier"] for tec in sets["tech"]["conversion"]], description = "carrier for conversion technology")
        carrier_out=Set(cep, name= "carrier_out", records =[techs[tec].output["carrier"] for tec in sets["tech"]["conversion"]], description = "carrier out for conversion technology")
        eff=Parameter(cep, name= "eff", domain=[tech_conv], records =np.array(pd.DataFrame(data=[techs[tec].constraints["efficiency"] for tec in sets["tech"]["conversion"]])), description = "efficiency for conversion technology")
        # Calculate Variable Costs
        e12=Equation(container=cep, name="e12", description=" variable cost for conversion technology")
        e12=COST["var", impact, tech_conv]==sign_generation* Sum((t,k,node), 
                                                                  GEN[tech_conv,carrier_conv , t,k,node ]
                                                                  *ts_weights[k]*ts_deltas[t,k]*costs[tech_conv, node, year, "var", impact])*scale[':GEN']/scale[':COST']
        # Calculate Fixed Costs
        e13=Equation(container=cep, name="e13", description="Fixed Costs for conversion technology")
        e13=COST["cap_fix",impact, tech_conv]==Sum((t,k), 
                                                    ts_weights[k]*ts_deltas[t,k] )/8760* Sum(node,
                                                                                             CAP[tech_conv,"new",node]*costs[tech_conv,node, year,"cap_fix",impact])*scale[':CAP']/scale[':COST']
        
        ## CONVERSION GEN ##
        # Limit the generation of conversion to the infrastructing capacity of conversion power plants
        e14=Equation(container=cep, name="e14")
        e15=Equation(container=cep, name="e15")
        e16=Equation(container=cep, name="e16")
        e14=GEN[tech_conv, carrier_conv, t, k, node]<=0
        e15=GEN[tech_conv, carrier_conv, t, k, node]>= (-1)*Sum(infrastruct, 
                                                                   CAP[tech_conv, infrastruct, node])*scale[':CAP']/scale[':GEN']
                                   
        e16= GEN[tech_conv,carrier_out, t , k, node]==(-1)*eff[tech_conv]*GEN[tech_conv,carrier_conv, t , k, node]                                       
    
    if config["storage"]:
        ## Set for storage
        tech_stor=Set(cep, name= "tech_stor", records =sets["tech"]["storage"], description = "storage technology")
        carrier_stor=Set(cep, name= "carrier_stor", records =[sets["carrier"][tec] for tec in sets["tech"]["storage"] ], description = "carrier for storage technology")
        effs=Parameter(cep, name= "effs", domain=tech_stor, records =np.array(pd.DataFrame([techs[tec].constraints["efficiency"] for tec in sets["tech"]["storage"]])), description = "efficiency for storage technology")
        ## VARIABLE
        STOR = Variable(
        container=cep,
        name="STOR",
        domain=[tech_stor, carrier_stor, t, k,node ],
        type="free",
        description="storage level",)
        
        ## STORAGE COSTS ##
        # Calculate variable costs
        e17=Equation(container=cep, name="e17", description=" variable cost for storage technology")
        e17=COST["var", impact, tech_stor]==0
        # Calculate fixed costs
        e18=Equation(container=cep, name="e18", description=" fixed cost for storage technology")
        e18=COST["cap_fix", impact, tech_stor]==Sum((t,k), 
                                                    ts_weights[k]*ts_deltas[t,k] )/8760* Sum(node,
                                                                                             CAP[tech_stor,"new",node]*costs[tech_stor,node, year,"cap_fix",impact])*scale[':CAP']/scale[':COST']
        ## STORAGE LEVEL ##
        e19=Equation(container=cep, name="e19", description="constraint-cyclic storage level")
        e19=STOR[tech_stor, carrier_stor,t, k, node]== STOR[tech_stor, carrier_stor, t.lead(1), k, node]*gams_math.power(effs[tech_stor], (ts_deltas[t,k]/732)) - ts_deltas[t,k]*GEN[tech_stor, carrier_stor,t, k, node]*scale[':GEN']/scale[':INTRASTOR']
        # Limit the storage of the energy part to its installed power
        e20=Equation(container=cep, name="e20", description="limit energy part of battery to its installed power ")
        e20= STOR[tech_stor, carrier_stor,t, k, node]<= Sum(infrastruct,
                                                            CAP[tech_stor,infrastruct, node]
                                                            )*scale[':CAP']/scale[':INTRASTOR']
        # Storage level at the beginning of each representative day equal
        e21=Equation(container=cep, name="e21")
        e21= STOR[tech_stor, carrier_stor, '0',k, node]== STOR[tech_stor,carrier_stor, '24', k, node]
        e22= Equation(container=cep, name="e22", description="Set storage level at the beginning of each representative day to the same")
        e22= STOR[tech_stor, carrier_stor,'0',k, node]==STOR[tech_stor, carrier_stor, '0', '1', node]
        
    # Fix designed variable to the installed caopacities if provided 
    if "fixed_design_variables" in config:
        fixed_design_variables=config["fixed_design_variables"]
        cap=Parameter(cep, name= "cap", domain=[tech_node, node], records =fixed_design_variables["CAP"], description = "fixed capacities found by expansion model")
        #TODO: add constraint line based
        #Constraint node based
        e23=Equation(container=cep, name="e23", description="Fix designed variable to the capacities") # domain should be verified
        e23=CAP[tech_node,"new",node]== cap/scale[':CAP']
        # TODO: check the value of "fixed_design_variables" to match the domain specified in "cap"
        
    if "fixed_design_variables" not in config:
        e24=Equation(container=cep, name="e24") 
        for tec in sets["tech"]["all"]:
            if "cap_eq" in techs[tec].constraints:
             # with 4 hours of storage, set this value for 'bat_e' to '4.0' (assuming hourly delta t).
                if "cap_eq_multiply" in techs[tec].constraints:
                 cap_eq_multiply= techs[tech].constraints["cap_eq_multiply"]
                else:
                 cap_eq_multiply=1.0
                e24=CAP[tec,"new",node]== cap_eq_multiply*CAP[techs[tec].constraints["cap_eq"],"new",node] # not correctly formatted
                     
    # Adding existing infrastructure to the model from the node table
    data_path2 = os.path.join(os.getcwd(), "data", config['region'])
    nodes_tech=load_cep_data_nodes_2(data_path2,techs)
    tech_ex=Set(cep, name= "tech_ex", records =set(sets["tech"]["node"]).intersection(sets["tech"]["exist_inf"]), description = "existing technology")
    ex={}
    for tecx in set(sets["tech"]["node"]).intersection(sets["tech"]["exist_inf"]):
        for n in sets['nodes']['all']:
            ex[tecx, n]=nodes_tech[tecx, n].power_ex
            
    ex_df = pd.DataFrame([(key[0], key[1], value) for key, value in ex.items()], columns=['tech', 'node', 'value'])
    EX=Parameter(cep, name= "EX", domain=[tech_ex, node], records =ex_df, description = "existing capacity of each each technology at each node")
 
    e24=Equation(container=cep, name="e24")
    e24=CAP[tech_ex, "ex", node]==EX[tech_ex,node]/scale[':CAP']
    
    # Limoit the infratructure expansion  
    #TODO 
    
    ## ENERGY BALANCE CONSTRAINT ##
    tech_e=Set(cep, name="tech_e", records=sets["tech"]["electricity"])
    tech_bat=Set(cep, name="tech_bat", records=sets["tech"]["electricity_bat"])
    e25=Equation(container=cep, name="e25", domain=["electricity",t, k, node], description="Energy balance constraints for carrier electricity")
    e26=Equation(container=cep, name="e26", domain=["electricity_bat",t, k, node], description="Energy balance constraints for carrier electricity_bat")
    e25=Sum(tech_e, GEN[tech_e, "electricity", t, k,node])==0
    e26=Sum(tech_bat, GEN[tech_bat, "electricity_bat", t, k,node])==0
    
    ## SETUP OBJECTIVE ##
    e27=Equation(container=cep, name="e27", description="Objective function")
    e27 = Sum([account, tech],
              COST[account, "EUR", tech])

    ## SOLVE MODEL##
    #Expansion = Model(cep, name="Expansion", problem="LP", sense="min", objective=e27)
    #Expansion.solve()
    #Expansion.num_domain_violations()
    
      
            
        
            
    
    
                                              
    print(cep.isValid())
    #carrier=techs["demand"]
    #carrier1=techs[tech_demand][dir_main_carrier]['carrier']
    #carrier2=getattr(techs["demand"], dir_main_carrier)["carrier"]
    #print(carrier2)
        
        
        
        
    
    
    #print(cep.listSymbols())
    
    #Links: https://gamspy.readthedocs.io/en/latest/user/notebooks/blend.html; https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html#pandas.DataFrame.from_dict 
    # https://stackoverflow.com/questions/55890752/creating-dataframe-from-a-dictionary-where-values-of-the-dict-are-numpy-array 
    
    
def reformat_df(dataframe):
    return dataframe.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Value"
    )
    
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
