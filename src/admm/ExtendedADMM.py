import numpy as np
from .SimpleADMM import SimpleADMM

class ExtendedADMM(SimpleADMM):
    # main loop with adaptive penalty parameter (rho)
    def solve(self, inc: float, dec: float, mu: float):
        """
        Solves the P2P trading problem based upon the given agents

        Parameters
        ----------
        inc: float
            penalty parameter is multiplied by parameter 'inc' if primal residual > mu * dual residual

        dec: float
            penalty parameter is divided by parameter 'dec' if dual residual > mu * primal residual

        mu: float
            decides sensitivity of penalty parameter (see 'inc' / 'dec' parameter definitions)
        """
        
        for k in range(self.max_iter):
            self.pi_prev = self.pi_mat.copy()
            self.update_primal()
            self.update_consensus()
            self.update_dual_lambda()
            self.update_dual_tau()
            self.update_dual_phi()

            primal_res = self.primal_residual()
            dual_res = self.dual_residual()

            self.primal_residual_history.append(primal_res)
            self.dual_residual_history.append(dual_res)

            if self.has_converged():
                break
            
            # adaptive penalty parameter step
            if primal_res > (mu * dual_res):
                self.rho = self.rho * inc
            elif dual_res > (mu * primal_res):
                self.rho = self.rho / dec
            # otherwise, keep the same

        return {
            "P": self.P_mat,
            "pi": self.pi_mat,
            "lambda": self.lambda_mat,
            "tau": self.tau,
            "phi": self.phi,
            "iterations": k
        }