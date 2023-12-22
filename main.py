
import os
import gurobipy as gp
from gurobipy import GRB
from utils.load_data import load_timeseries_data_provided, load_cep_data_techs, load_cep_data_lines, load_cep_data_costs, load_cep_data_nodes, load_cep_data, load_cep_data_provided
from opt_problems.run_opt import run_opt

region = "GER_2"
data_path = os.path.join(os.getcwd(), "data", region)

optimizer = gp
ts_input_data = load_timeseries_data_provided(region, T=24, years=[2016])#
cep_data=load_cep_data(data_path, region)
cep_result = run_opt(ts_input_data, cep_data, optimizer, descriptor="simple storage", transmission=True, storage_type="simple", conversion=True)
#print("End")
