import numpy as np
from src.admm.SimpleADMM import SimpleADMM
from src.admm.ExtendedADMM import ExtendedADMM
from src.agents.Consumer import Consumer
from src.agents.Producer import Producer
from src.agents.Grid import Grid
from src.agents.Prosumer import Prosumer

def build_agents():
    return [
        Producer(0, 0, 5, 20, 0.5, 3),
        Producer(1, 1, 5, 20, 0.6, 5),

        Consumer(2, 2, 5, 20, 1, 12),
        Consumer(3, 3, 5, 20, 1, 15),
    ]

def print_log(log):
    """
    Helper function to print results from ADMM.solve() function
    """
    P = log['P']
    print("Trades")
    print(P)
    print("---")

    print("# Iterations")
    print(log["iterations"])
    print("---")

    print("\nGlobal imbalance sum(P):", np.sum(P))

    
def main():

    np.set_printoptions(precision=4, suppress=True)

    for rho in [0.1, 0.5, 1, 2, 5, 10]:
        print("======================================")
        print("Simulation (rho = " + str(rho) + ")")
        print("---")
        simple_model = SimpleADMM(
            agents=build_agents(),
            rho=rho,
            max_iter=1000,
            tol=1e-4
        )

        print("Running SimpleADMM")
        print("---")

        log = simple_model.solve()   # <- clean run, no history, no noise
        print_log(log)

        print("=====================================")

        extended_model = ExtendedADMM(
            agents=build_agents(),
            rho=rho,
            max_iter=1000,
            tol=1e-4
        )

        print("Running ExtendedADMM")
        print("---")

        log = extended_model.solve(2,2,10)   # <- clean run, no history, no noise
        print_log(log)


if __name__ == "__main__":
    main()