from .Agent import Agent
import numpy as np

class Grid(Agent):
    """
    External Grid Agent

    Attributes
    ----------
        agent_id : int
            Agent's unique identifier
        max_import: float
            Maximum power which the grid can import (kW)
        max_export: float
            Maximum power which the agent can export (kW)
        buy_price: float
            price to purchase energy from grid, ($/kWh)
        sell_price: float
            price grid will sell energy for ($/kWh)
    """

    def __init__(self, agent_id: int, max_import: float, max_export: float, buy_price: float, sell_price: float):
        super().__init__(agent_id, 0, -max_import, max_export)

        self.buy_price = buy_price
        self.sell_price = sell_price


    def solve_local(self, P_i, pi_row, lambda_row, rho, tau, phi):
        """
        Solve the local optimization problem for the grid agent in an ADMM-based energy trading setup.

        Parameters
        ----------
        P_i : numpy.ndarray
            Current power vector for agent i (including self entry).
        pi_row : numpy.ndarray
            Consensus variables π_i, pi_row[j] = π_{i, j}
        lambda_row : numpy.ndarray
            Dual variables λ_i for ADMM coupling terms. lambda_row[j] = λ_{i,j}
        rho : float
            ADMM penalty parameter.
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


