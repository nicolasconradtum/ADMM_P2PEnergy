from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    """
    Abstract Agent Class
    """
    
    def __init__(self, agent_id: int, bus: int, p_min:float=0.0, p_max:float=0.0):
        self.agent_id = agent_id
        self.bus_id = bus

        self.p_min = p_min
        self.p_max = p_max

    @abstractmethod
    def solve_local(self, P_i: np.ndarray, pi_row: np.ndarray, lambda_row: np.ndarray, rho: float, tau: float, phi: float):
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
        tau : float
            Dual variable for upper power constraint
        phi : float
            Dual variable for lower power constraint

        Returns
        -------
        P_opt : numpy.ndarray
            Optimized power vector for agent i with self-trade enforced to zero.
        """
        pass