import numpy as np

from src.agents.Consumer import Consumer
from src.agents.Producer import Producer
from src.agents.Prosumer import Prosumer

from src.admm.AdaptivePenaltyADMM import AdaptivePenaltyADMM
from src.admm.FastADMM import FastADMM

from src.grid.GridManagerPP import GridManagerPP

import pandapower as pp
from pandapower.networks import create_kerber_landnetz_freileitung_1, create_kerber_landnetz_freileitung_2

# helper function for printing matrix results
def matprint(mat, fmt="g"):
    mat = np.array(mat)

    # If it's a vector (1D), reshape to (n, 1) for uniform handling
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)

    col_maxes = [
        max(len(("{:" + fmt + "}").format(x)) for x in col)
        for col in mat.T
    ]

    for row in mat:
        for i, y in enumerate(row):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print()

net1 = create_kerber_landnetz_freileitung_2()
net2 = create_kerber_landnetz_freileitung_2()

agents1 = [
    Consumer(agent_id=0, p_min=-35, p_max=-30, utility_a=0.9, utility_b=13, power_factor=1),
    Consumer(agent_id=1, p_min=-20, p_max=-15, utility_a=1.25, utility_b=9.1, power_factor=1),
    Consumer(agent_id=2, p_min=-20, p_max=-12, utility_a=1, utility_b=16, power_factor=1),
    Consumer(agent_id=3, p_min=-25, p_max=-20, utility_a=1.1, utility_b=12, power_factor=1),
    Producer(agent_id=4, p_min=2, p_max=30, cost_a=0.26,cost_b=6, power_factor=1),
    Producer(agent_id=5, p_min=2, p_max=40, cost_a=0.4, cost_b=2.6, power_factor=1),
    Producer(agent_id=6, p_min=5, p_max=30, cost_a=0.6, cost_b=4, power_factor=1),
    Producer(agent_id=7, p_min=0, p_max=20, cost_a=0.7, cost_b=6.5, power_factor=1)
]

agents2 = [
    Consumer(agent_id=0, p_min=-35, p_max=-30, utility_a=0.9, utility_b=13, power_factor=1),
    Consumer(agent_id=1, p_min=-20, p_max=-15, utility_a=1.25, utility_b=9.1, power_factor=1),
    Consumer(agent_id=2, p_min=-20, p_max=-12, utility_a=1, utility_b=16, power_factor=1),
    Consumer(agent_id=3, p_min=-25, p_max=-20, utility_a=1.1, utility_b=12, power_factor=1),
    Producer(agent_id=4, p_min=2, p_max=30, cost_a=0.26,cost_b=6, power_factor=1),
    Producer(agent_id=5, p_min=2, p_max=40, cost_a=0.4, cost_b=2.6, power_factor=1),
    Producer(agent_id=6, p_min=5, p_max=30, cost_a=0.6, cost_b=4, power_factor=1),
    Producer(agent_id=7, p_min=0, p_max=20, cost_a=0.7, cost_b=6.5, power_factor=1)
]

gm1 = GridManagerPP(net1, agents1)
gm2 = GridManagerPP(net2, agents2)

ADMM_Solver1 = FastADMM(agents=agents1, grid_manager=gm1, enforce_grid_constraints=False, rho=2, max_iter=500, tol=1e-3)
ADMM_Solver2 = FastADMM(agents=agents2, grid_manager=gm2, enforce_grid_constraints=True, rho=2, max_iter=500, tol=1e-3)

log_unconstrained = ADMM_Solver1.solve(inc=2, dec=2, mu=10)
log_constrained = ADMM_Solver2.solve(inc=2, dec=2, mu=10)

print("====== # ITERATIONS ======")
print("F-ADMM Unconstrained: " + str(log_unconstrained['iterations']))
print("F-ADMM Constrained: " + str(log_constrained['iterations']) +  "\n")

print("====== TRADE MATRIX ======")
print("F-ADMM Unconstrained: ")
print("---")
matprint(log_unconstrained['P'])
print("\n")
print("--------------------------")
print("F-ADMM Constrained: ")
print("---")
matprint(log_constrained['P'])
print("\n")
print("====== NET INJECTIONS PER AGENT ======")
print("F-ADMM Unconstrained: ")
print("---")
net_injections_1 = np.sum(log_unconstrained['P'], axis=1)
matprint(net_injections_1)
print("\n")
print("F-ADMM Constrained: ")
print("---")
net_injections_2 = np.sum(log_constrained['P'], axis=1)
matprint(net_injections_2)
print("\n")
print("====== LinDistFlow ======")

print("F-ADMM Unconstrained: ")
print("---")
v1, P_flow1, Q_flow1 = gm2.compute_lindistflow(log_unconstrained['P'])
print("1. Bus Voltages (p.u.)")
matprint(np.concatenate(([1], v1)))
print("\n")
print("-------------------------")
print("2. Real Power Flow (kW)")
matprint(P_flow1 * gm2.S_base * 1000)
print("\n")
print("-------------------------")
print("2. Reactive Power Flow (kW)")
matprint(Q_flow1 * gm2.S_base * 1000)
print("-------------------------")

print("F-ADMM Constrained: ")
print("---")
v2, P_flow2, Q_flow2 = gm2.compute_lindistflow(log_constrained['P'])
print("1. Bus Voltages (p.u.)")
matprint(np.concatenate(([1], v2)))
print("\n")
print("-------------------------")
print("2. Real Power Flow (kW)")
matprint(P_flow2 * gm2.S_base * 1000)
print("\n")
print("-------------------------")
print("2. Reactive Power Flow (kW)")
matprint(Q_flow2 * gm2.S_base * 1000)
print("-------------------------")
print("\n")
print("====== Full AC power flow ======")
net1.load["p_mw"] = 0.0
net2.load["p_mw"] = 0.0


print("F-ADMM Unconstrained: ")
for i, row in net1.load.iterrows():
    bus = row["bus"]
    net1.load.at[i, "p_mw"] = -(net_injections_1[i] / 1000)

pp.runpp(net1)

print("Converged: " + str(net1.converged))
print("Bus Voltages (p.u.) ")
matprint(net1.res_bus["vm_pu"])
print("Line Flows (kW) ")
matprint(net1.res_line[["p_from_mw", "p_to_mw"]] * 1000)
print("----------------------------")
print("F-ADMM Constrained: ")
for i, row in net2.load.iterrows():
    bus = row["bus"]
    net2.load.at[i, "p_mw"] = -(net_injections_2[i] / 1000)

pp.runpp(net2)

print("Converged: " + str(net2.converged))
print("Bus Voltages (p.u.) ")
matprint(net2.res_bus["vm_pu"])
print("Line Flows (kW) ")
matprint(net2.res_line[["p_from_mw", "p_to_mw"]] * 1000)

