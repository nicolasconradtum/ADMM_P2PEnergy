from abc import ABC, abstractmethod
import numpy as np
from src.grid.GridManager import GridManager
from src.agents.Agent import Agent

class ADMM(ABC):
    def __init__(self, agents: list[Agent], grid_manager: GridManager, enforce_grid_constraints: bool, rho=1.0, max_iter=500, tol=1e-4):

        self.agents = agents
        self.N = len(agents)

        self.grid_manager = grid_manager
        self.enforce_grid_constraints = enforce_grid_constraints

        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol

        # primal variables
        self.P_mat = np.zeros((self.N, self.N))

        # consensus variables
        self.pi_mat = np.zeros((self.N, self.N))
        self.pi_prev = np.zeros((self.N, self.N))

        # dual variables
        self.lambda_mat = np.zeros((self.N, self.N))

        # history
        self.primal_residual_history = []
        self.dual_residual_history = []

    # residuals
    def primal_residual(self):
        return np.linalg.norm(self.pi_mat - self.P_mat)

    def dual_residual(self):
        return self.rho * np.linalg.norm(self.pi_mat - self.pi_prev)

    def has_converged(self):

        return (
            self.primal_residual() < self.tol
            and
            self.dual_residual() < self.tol
        )

    # abstract steps
    @abstractmethod
    def update_primal(self):
        pass

    @abstractmethod
    def update_consensus(self):
        pass

    @abstractmethod
    def update_dual_lambda(self):
        pass

    def solve(self):
        """
        Solves the P2P trading problem based upon the given agents
        """
        for k in range(self.max_iter):
            self.pi_prev = self.pi_mat.copy()
            self.update_primal()
            self.update_consensus()
            self.update_dual_lambda()

            for i, agent in enumerate(self.agents):
                p_i = np.sum(self.P_mat[i,:])
                agent.update_dual_tau_p(p_i)
                agent.update_dual_phi_p(p_i)


            if self.enforce_grid_constraints:
                self.grid_manager.update_dual_grid(self.P_mat)

            self.primal_residual_history.append(self.primal_residual())
            self.dual_residual_history.append(self.dual_residual())

            if self.has_converged():
                break

        return {
            "P": self.P_mat,
            "pi": self.pi_mat,
            "lambda": self.lambda_mat,
            "iterations": k
        }