import numpy as np
import pandapower as pp
from src.agents.Agent import Agent

class GridManagerPP:
    def __init__(self, net: pp.pandapowerNet, agents: list[Agent]):
        self.net = net
        self.agents = agents

        self.buses = net.bus.index.to_list()
        self.lines = net.line.index.to_list()
        self.num_lines = len(net.line)

        self.lv_buses = net.bus[net.bus.vn_kv == net.bus.vn_kv.min()].index
        first_lv_bus = self.lv_buses[0]

        # reindex: first distribution bus becomes index 0
        self.bus_to_idx = {
            bus: i for i, bus in enumerate(self.lv_buses)
        }

        self.num_buses = len(self.lv_buses)

        # ensure number of agents = number of buses
        if len(agents) != self.num_buses - 1:
            raise ValueError(
                f"Number of agents ({len(agents)}) does not match "
                f"number of model buses ({self.num_buses}). "
                f"Expected one agent per non-slack bus."
            )

        # base units
        self.S_base = 100 # MVA
        self.V_base = net.bus.loc[first_lv_bus, "vn_kv"] # kV
        self.Z_base = (self.V_base ** 2) / self.S_base # ohms

        # build network matrices
        self.M_full = self._build_incidence_matrix()
        self.M_reduced = self._build_reduced_incidence_matrix()
        self.Dr, self.Dx = self._build_diagonal_line_params()
        self.A = self._compute_A()
        self.R = self._compute_R()
        self.X = self._compute_X()

        # initialize dual variables
        self.tau_v = np.zeros(self.num_buses - 1)
        self.phi_v = np.zeros(self.num_buses - 1)

        self.tau_s = np.zeros(self.num_lines)
        self.phi_s = np.zeros(self.num_lines)

        self.dual_step_size = {'v': 0.05, 's': 0.05}

    # build full incidence matrix
    def _build_incidence_matrix(self):
        M = np.zeros((self.num_lines, self.num_buses))

        for l, line in self.net.line.iterrows():
            f_bus = int(line.from_bus)
            t_bus = int(line.to_bus)

            f = self.bus_to_idx[f_bus]
            t = self.bus_to_idx[t_bus]

            M[l, f] = 1
            M[l, t] = -1

        return M

    def _build_reduced_incidence_matrix(self):
        return self.M_full[:,1:]

    # line resistance / reactance matrices
    def _build_diagonal_line_params(self):
        r = (self.net.line["r_ohm_per_km"] * self.net.line["length_km"]).to_numpy()
        x = (self.net.line["x_ohm_per_km"] * self.net.line["length_km"]).to_numpy()

        r_pu = r / self.Z_base
        x_pu = x / self.Z_base

        return np.diag(r_pu), np.diag(x_pu)

    # LinDistFlow Matrices
    def _compute_A(self):
        return np.linalg.inv(self.M_reduced).T

    def _compute_R(self):
        M_inv = np.linalg.inv(self.M_reduced)
        return M_inv @ self.Dr @ M_inv.T

    def _compute_X(self):
        M_inv = np.linalg.inv(self.M_reduced)
        return M_inv @ self.Dx @ M_inv.T

    # LinDistFlow
    def compute_lindistflow(self, P_mat, v0=1.0):
        power_factors = np.zeros(self.num_buses - 1)

        for bus in self.lv_buses:
            i = self.bus_to_idx[bus]
            if i == 0: 
                continue

            power_factors[i-1] = self.agents[i-1].power_factor

        kappa = np.tan(np.arccos(power_factors))

        # net injections (kW)
        p_kW = np.sum(P_mat, axis=1)
        p = p_kW / (1000 * self.S_base) # factor 1000 for proper scaling

        q = p * kappa

        # line flows
        P_flow = self.A @ p
        Q_flow = self.A @ q

        # voltages
        v = v0 + self.R @ p + self.X @ q

        return v, P_flow, Q_flow

    # Grid Dual Updates
    def update_dual_grid(self, P_mat):
        v, P_flow, Q_flow = self.compute_lindistflow(P_mat)
        P_flow_kW = P_flow * self.S_base * 1000 # convert pu power flows to kW

        v_max, v_min = 1.05, 0.95
        s_max, s_min = 40, -40

        self.tau_v = np.maximum(0, self.tau_v + self.dual_step_size['v'] * (v - v_max))
        self.phi_v = np.maximum(0, self.phi_v + self.dual_step_size['v'] * (v_min - v))
        self.tau_s = np.maximum(0, self.tau_s + self.dual_step_size['s'] * (P_flow_kW - s_max))
        self.phi_s = np.maximum(0, self.phi_s + self.dual_step_size['s'] * (s_min - P_flow_kW))

    # calculate grid coupling term
    def compute_G_i(self, i):
        
        # kappa = np.tan(np.arccos(power_factor))
        v_term = np.sum(self.R[:, i] * (self.tau_v - self.phi_v))
        s_term = np.sum(self.A[:, i] * (self.tau_s - self.phi_s))

        return v_term + s_term