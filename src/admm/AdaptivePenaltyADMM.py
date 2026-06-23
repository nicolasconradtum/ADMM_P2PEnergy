import numpy as np
from .SimpleADMM import SimpleADMM

class AdaptivePenaltyADMM(SimpleADMM):
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

            # update local dual variables in agent
            for i, agent in enumerate(self.agents):
                p_i = np.sum(self.P_mat[i,:])
                agent.update_dual_tau_p(p_i)
                agent.update_dual_phi_p(p_i)
                
            # update grid constraint dual variables in grid manager
            if self.enforce_grid_constraints:
                self.grid_manager.update_dual_grid(self.P_mat)

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
            "iterations": k
        }