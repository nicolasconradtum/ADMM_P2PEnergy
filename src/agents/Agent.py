from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    """
    Agent class representing a market participant in the
    peer-to-peer energy trading framework.

    Attributes
    ----------
    agent_id : int
        Unique identifier of the agent.

    p_min : float
        Minimum active power limit in kW (Negative for consumption,
        positive for production)

    p_max : float
        Maximum active power limit in kW (Negative for consumption,
        positive for production)

    dual_step_size : float
        Step size used for updating agent's local dual variables during optimization.

    power_factor : float
        Agent power factor used to compute reactive power injections or
        withdrawals from active power values.
    """
    def __init__(self, agent_id: int, p_min:float, p_max:float, power_factor: float):
        # ensure p_max ≥ p_min
        if p_max < p_min:
            raise ValueError(
                f"Invalid power limits: p_min ({p_min}) must be ≤ p_max ({p_max})"
            )
        
        self.agent_id = agent_id
        self.p_min = p_min
        self.p_max = p_max 
        self.tau_p = 0 # dual variable for lower power constraint
        self.phi_p = 0 # dual variable for upper power constraint
        self.dual_step_size = 0.1
        self.power_factor=power_factor


    @abstractmethod
    def solve_local(self, P_i: np.ndarray, pi_row: np.ndarray, lambda_row: np.ndarray, rho: float, G_i: float):
        """
        Solve the local optimization problem for an agent in an ADMM-based energy trading setup.

        Parameters
        ----------
        P_i : ndarray
            Current power vector for agent i (including self entry).
        pi_row : ndarray
            Consensus variables π_i, pi_row[j] = π_{i, j}
        lambda_row : ndarray
            Dual variables λ_i for ADMM coupling terms. lambda_row[j] = λ_{i,j}
        rho : float
            ADMM penalty parameter
        tau : dict
            Dual variable for upper power, voltage, line flow constraints
        phi : float
            Dual variable for lower power, voltage, line flow constraints

        Returns
        -------
        P_opt : numpy.ndarray
            Optimized power vector for agent i with self-trade enforced to zero.
        """
        pass

    # dual update - tau_p (upper agent power constraint)
    def update_dual_tau_p(self, p_i):
        self.tau_p = max(0.0, self.tau_p + self.dual_step_size * (p_i - self.p_max))

    # dual update - phi_p (lower agent power constraint)
    def update_dual_phi_p(self, p_i):
        self.phi_p = max(0.0, self.phi_p + self.dual_step_size * (self.p_min - p_i))

    def get_p_max(self):
        return self.p_max
    
    def get_p_min(self):
        return self.p_min