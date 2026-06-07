from abc import ABC, abstractmethod
import numpy as np

class ADMM(ABC):
    def __init__(self, agents, rho=1.0, max_iter=500, tol=1e-4):
        self.agents = agents

        self.N = len(agents)

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
        self.tau = np.zeros(self.N)
        self.phi = np.zeros(self.N)

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

    @abstractmethod
    def update_dual_tau(self):
        pass

    @abstractmethod
    def update_dual_phi(self):
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
            self.update_dual_tau()
            self.update_dual_phi()

            self.primal_residual_history.append(self.primal_residual())

            self.dual_residual_history.append(self.dual_residual())

            if self.has_converged():
                break

        return {
            "P": self.P_mat,
            "pi": self.pi_mat,
            "lambda": self.lambda_mat,
            "tau": self.tau,
            "phi": self.phi,
            "iterations": k
        }
