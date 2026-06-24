import numpy as np
import pandas as pd
import pandapower as pp
import networkx as nx
import pandapower.networks as pn

class GridModel:
    def __init__(self, net):
        # raw pandapower network
        self.net = net

        # system base power
        self.S_base_MVA = net.sn_mva 
        
        # dictionary: IEEE bus indices --> clean indices (0, 1, 2, ..., N)
        self.bus_id_to_idx = {bus_id: idx for idx, bus_id in enumerate(net.bus.index)}
        
        # dictionary: clean indices (0, 1, 2, ..., N) --> IEEE bus indices 
        self.bus_idx_to_id = {idx: bus_id for bus_id, idx in self.bus_id_to_idx.items()}
        
        # store how many buses and lines are in the grid
        self.num_buses = len(net.bus)
        self.num_lines = len(net.line)
        
        # find where the main utility grid connects (Substation / Slack Bus)
        slack_bus_id = net.ext_grid.bus.values[0]
        self.slack_bus_idx = self.bus_id_to_idx[slack_bus_id]

        # create an empty injection matrix
        # size = (# lines) x (# buses)
        self.A_matrix = np.zeros((self.num_lines, self.num_buses))

        # fill Matrix A with data from grid
        self._build_injection_matrix()

        # build Matrix Rsens (Active Power Voltage Map)
        self.R_sens = np.zeros((self.num_buses, self.num_buses))
        
        # build Matrix Xsens (Reactive Power Voltage Map)
        self.X_sens = np.zeros((self.num_buses, self.num_buses))
        
        # trigger the method to fill both sensitivity matrices with data
        self._build_voltage_sensitivity_matrices()


    def _build_injection_matrix(self):
        """
        Builds Matrix A cleanly using NetworkX graph paths.
        """
        # convert the pandapower line list directly into a NetworkX Graph
        graph = nx.Graph()

        # loop through all line
        for idx, line in self.net.line.iterrows():
            # add line to graph 
            u, v = self.bus_id_to_idx[line.from_bus], self.bus_id_to_idx[line.to_bus]
            graph.add_edge(u, v, line_idx=idx)
            
        # for every bus, find the unique path back to the substation (slack bus)
        for node_idx in range(self.num_buses):
            # ignore the slack bus
            if node_idx == self.slack_bus_idx:
                continue
                
            # get the exact sequence of nodes from this bus to the slack bus
            path = nx.shortest_path(graph, source=node_idx, target=self.slack_bus_idx)
            
            # for every pair of connected nodes in that path, mark the line in Matrix A
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                line_idx = graph[u][v]['line_idx']
                
                # check orientation: +1 if moving with pandapower's direction, -1 if against
                from_idx = self.bus_id_to_idx[self.net.line.loc[line_idx, "from_bus"]]
                self.A_matrix[line_idx, node_idx] = 1.0 if u == from_idx else -1.0


    def _build_voltage_sensitivity_matrices(self):
        """
        Builds both Matrix Rsens and Matrix Xsens using line parameters and Matrix A.
        """
        # create empty arrays to store per-unit resistances and reactances for all lines
        r_pu = np.zeros(self.num_lines)
        x_pu = np.zeros(self.num_lines)

        # loop through all line
        for idx, line in self.net.line.iterrows():
            # find nominal voltage where this line connects
            V_base_kV = self.net.bus.loc[line.from_bus, "vn_kv"]
            
            # calculate base impedance
            Z_base_ohm = (V_base_kV ** 2) / self.S_base_MVA
            
            # calculate absolute resistance Ohms by multiplying by length
            r_ohm = line.r_ohm_per_km * line.length_km
            
            # calculate absolute reactance Ohms by multiplying by length
            x_ohm = line.x_ohm_per_km * line.length_km
            
            # convert to Per-Unit and store in respective arrays
            r_pu[idx] = r_ohm / Z_base_ohm
            x_pu[idx] = x_ohm / Z_base_ohm

        # create diagonal matrices out of the resistance and reactance arrays
        R_diagonal = np.diag(r_pu)
        X_diagonal = np.diag(x_pu)
        
        # combine matrix A and parameters to map voltage sensitivities
        # formula: matrix = A_transpose * Diagonal_Matrix * A
        self.R_sens = np.dot(np.dot(self.A_matrix.T, R_diagonal), self.A_matrix)
        self.X_sens = np.dot(np.dot(self.A_matrix.T, X_diagonal), self.A_matrix)

    def calculate_line_flows(self, p, q):
        """
        Calculates both active and reactive power flows on lines in per-unit.

        Parameters
        ----------
        p: ndarray
            vector of bus injected active power
        q: ndarray
            vector of bus injected reactive power
        """
        # calculate active power line flows
        P_line = np.dot(self.A_matrix, p)
        # calculate reactive power line flows
        Q_line = np.dot(self.A_matrix, q)
        return P_line, Q_line   

    def calculate_bus_voltages(self, p, q, V_slack=1.0):
        """
        Calculates the per-unit voltage considering both active and reactive power

        Parameters
        ----------
        p: ndarray
            vector of bus injected active power
        q: ndarray
            vector of bus injected reactive power
        V_slack: float
            per unit (p.u.) slack bus voltage
        """
        # calculate the voltage changes caused by active power injections
        delta_V_active = np.dot(self.R_sens, p)
        
        # initialize an empty voltage change buffer for reactive power
        delta_V_reactive = np.zeros(self.num_buses)
        
        # calculate the voltage changes caused by reactive power injections
        delta_V_reactive = np.dot(self.X_sens, q)
            
        # add the active and reactive voltage shifts to the substation reference voltage
        bus_voltages = V_slack + delta_V_active + delta_V_reactive
        
        # return an array containing the completed voltage profile
        return bus_voltages
