from abc import ABC, abstractmethod
import numpy as np
from src.agents.Agent import Agent

class GridAgent(Agent):

    def __init__(self, import_price: float, export_price: float, **kwargs):
        super().__init__(**kwargs)
        self.import_price = import_price
        self.export_price = export_price

    def solve_local(self, P_i: np.ndarray, pi_row: np.ndarray, lambda_row: np.ndarray, rho: float, G_i: float):
        pass

    # dual update - tau_p (upper agent power constraint)
    def update_dual_tau_p(self, p_i):
        self.tau_p = max(0.0, self.tau_p + self.dual_step_size * (p_i - self.p_max))

    # dual update - phi_p (lower agent power constraint)
    def update_dual_phi_p(self, p_i):
        self.phi_p = max(0.0, self.phi_p + self.dual_step_size * (self.p_min - p_i))