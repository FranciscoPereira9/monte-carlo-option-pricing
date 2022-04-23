import random
import numpy as np
from math import e, log, sqrt, pi
from scipy.stats import norm
import matplotlib.pyplot as plt

seed = None

class Option:

    def __init__(self, S, K, T, r, sigma, style, type):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.style = style
        self.type = type

    def calculate_payoff(self, St):
        """
        Calculates payoff based on Option type.
        :param St: stock price values.
        :return: intrinsic payoff value.
        """
        if self.type == 'call':
            payoff = np.clip(St - self.K, 0, None)
        elif self.type == "put":
            payoff = np.clip(self.K - St, 0, None)
        else:
            print("ERROR: unrecognized specified -> self.type ...")
            payoff = None
        return payoff

    def calculate_digital_payoff(self, St):
        """
        Calculates Digital Payoff based on Option type.
        :param St: stock price values.
        :return: smoothen digital payoff.
        """
        payoff = self.calculate_payoff(St)
        # Digital Payoff
        payoff[payoff > 0] = 1
        return payoff

    def calculate_smooth_digital_payoff(self, St):
        """
        Calculates Digital Payoff based on Option type.
        :param St: stock price values.
        :return: smoothen digital payoff.
        """
        payoff = self.calculate_payoff(St)
        # Smoothen Step Function -> payoff[payoff > 0] = 1
        return payoff

    def binomial_asset_prices(self, M):
        """
        Simulate Asset Price in a two-state economy (binomial model) for N time steps.
        :param M: number of steps to consider
        :return: M by M matrix with binomial asset prices
        """
        dt = self.T / M
        matrix = np.zeros((M + 1, M + 1))
        u = e ** (self.sigma * np.sqrt(dt))
        d = e ** (-self.sigma * np.sqrt(dt))
        # Iterate over the lower triangle
        for i in np.arange(M + 1):
            for j in np.arange(i + 1):
                # Express each cell as a combination of up and down moves
                matrix[i, j] = self.S * (u ** j) * (d ** (i - j))
        return matrix

    def binomial_option_values(self, asset_prices):
        """
        Calculates binomial option values.
        :param asset_prices: M by M matrix with binomial asset prices
        :return: M by M matrix with binomial option prices
        """
        # Parameters
        dt = T / len(asset_prices)
        u = e ** (self.sigma * np.sqrt(dt))
        d = e ** (-self.sigma * np.sqrt(dt))
        p = ((e ** (r * dt)) - d) / (u - d)
        # Tree
        option_tree = np.zeros(asset_prices.shape)
        columns = asset_prices.shape[1]
        rows = asset_prices.shape[0]

        # Add pay-off function in the last row
        for c in np.arange(columns):
            option_tree[rows - 1, c] = self.calculate_payoff(asset_prices[rows - 1, c])

        # For other rows, combine with previous rows. Walk backwards, from last row to first row
        for i in np.arange(rows - 1)[::-1]:
            for j in np.arange(i + 1):
                down = option_tree[i + 1, j]
                up = option_tree[i + 1, j + 1]
                if self.style == "european":
                    option_tree[i, j] = (e ** (-r * dt)) * ((p * up) + ((1 - p) * down))
                elif self.style == "american":
                    option_tree[i, j] = max((e ** (-r * dt)) * ((p * up) + ((1 - p) * down)),
                                            self.calculate_payoff(asset_prices[i, j]))
                else:
                    print("ERROR: wrong specified parameters ...")
        return option_tree

    def blackscholes_option_value(self):
        if self.type == 'call':
            option_value = self.bs_call()
        elif self.type == "put":
            option_value = self.bs_put()
        else:
            print("ERROR: unrecognized specified -> self.type ...")
            option_value = None
        return option_value

    def d1(self):
        return (log(self.S / self.K) + (self.r + self.sigma ** 2 / 2.) * self.T) / (self.sigma * sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * sqrt(self.T)

    def bs_call(self):
        return self.S * norm.cdf(self.d1()) - (self.K * e ** (-self.r * self.T) * norm.cdf(self.d2()))

    def bs_put(self):
        return self.K * e ** (-self.r * self.T) - self.S + self.bs_call()

    def bs_delta(self):
        if self.type == "call":
            delta = norm.cdf(self.d1())
        elif self.type == "put":
            delta = norm.cdf(self.d1()) - 1
        else:
            print("ERROR: unrecognized specified -> self.type ...")
            delta = None
        return delta

    def monte_carlo_simulation(self, M=1, n_runs=20, viz=False, seed=None):
        """
        Performs Monte Carlo simulations for an asset price.
        :param M: time steps
        :param n_runs: number of simulations
        :param viz: boolean. True to visualize the simulations.
        :return: [M, n_runs] Matrix of asset paths
        """
        dt = T / M
        # Simulate prices
        if seed is not None:
            np.random.seed(seed)
        M, n_runs = int(M), int(n_runs)
        ln_st = np.log(self.S) + np.cumsum(((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) *
                                            np.random.normal(size=(M, n_runs))), axis=0)
        simulation_matrix = np.exp(ln_st)

        if viz:
            plt.plot(simulation_matrix);
            plt.xlabel("Time Steps")
            plt.ylabel("Asset Price")
            plt.title("Monte Carlo Simulations")
            plt.show()

        return simulation_matrix

    def monte_carlo_option_value(self, simulation_matrix):
        # Steps and Runs
        M = simulation_matrix.shape[0]
        n_runs = simulation_matrix.shape[1]
        # Calculate payoff and discount back
        if self.style == "european":
            # Calculate Payoff at maturity
            payoffs_t = self.calculate_payoff(simulation_matrix[-1, :])
            # Discount Back
            option_values = payoffs_t * e ** (-r * T)
            option_value, std_error = option_values.mean(),  option_values.std()/np.sqrt(n_runs)
        elif self.style == "american":
            # TODO
            print("Not implemented yet ...")
            option_value, std_error = None, None
        elif self.style == "asian":
            # TODO
            print("Not implemented yet ...")
            option_value, std_error = None, None
        else:
            print("ERROR: unrecognized specified -> self.type ...")
            option_value, std_error = None, None
        return option_value, std_error

    def monte_carlo_option_value_digital(self, simulation_matrix):
        # Steps and Runs
        M = simulation_matrix.shape[0]
        n_runs = simulation_matrix.shape[1]
        # Calculate payoff and discount back
        if self.style == "european":
            # Calculate Payoff at maturity
            payoffs_t = self.calculate_digital_payoff(simulation_matrix[-1, :])
            # Discount Back
            option_values = payoffs_t * e ** (-r * T)
            option_value, std_error = option_values.mean(),  option_values.std()/np.sqrt(n_runs)
        elif self.style == "american":
            # TODO
            print("Not implemented yet ...")
            option_value, std_error = None, None
        elif self.style == "asian":
            # TODO
            print("Not implemented yet ...")
            option_value, std_error = None, None
        else:
            print("ERROR: unrecognized specified -> self.type ...")
            option_value, std_error = None, None
        return option_value, std_error

    def bump_and_revalue_delta(self, simulation_matrix, epsilon):
        M, n_runs = simulation_matrix.shape[0], simulation_matrix.shape[1]
        # Bump and Revalue method
        S_aux = self.S
        self.S += epsilon
        shocked_up = self.monte_carlo_simulation(M, n_runs, seed=seed)
        self.S = S_aux
        # Option values
        option_value, std_error = self.monte_carlo_option_value(simulation_matrix)
        shocked_option_value, shocked_std_error = self.monte_carlo_option_value(shocked_up)
        delta = (shocked_option_value - option_value)/epsilon
        delta_std_error = std_error + shocked_std_error
        return delta, delta_std_error

    def bump_and_revalue_delta_digital(self, simulation_matrix, epsilon):
        M, n_runs = simulation_matrix.shape[0], simulation_matrix.shape[1]
        # Bump and Revalue method
        S_aux = self.S
        self.S += epsilon
        shocked_up = self.monte_carlo_simulation(M, n_runs, seed=seed)
        self.S = S_aux
        # Option values
        option_value, std_error = self.monte_carlo_option_value_digital(simulation_matrix)
        shocked_option_value, shocked_std_error = self.monte_carlo_option_value_digital(shocked_up)
        delta = (shocked_option_value - option_value)/epsilon
        delta_std_error = std_error + shocked_std_error
        return delta, delta_std_error


    def pathwise_method(self, simulation_matrix, std = None):
        # Calculate digital payoffs
        payoffs_t = self.calculate_smooth_digital_payoff(simulation_matrix[-1, :])
        delta = e ** (-self.r * self.T) * norm.pdf(payoffs_t, payoffs_t.mean(), payoffs_t.std()) * simulation_matrix[-1, :] / self.S
        if std:
            delta = e ** (-self.r * self.T) * norm.pdf(payoffs_t, payoffs_t.mean(),
                                                       std) * simulation_matrix[-1, :] / self.S
        return delta


    def likelihood_ratio_method(self, simulation_matrix):
        # Calculate digital payoffs
        if seed is not None:
            np.random.seed(seed)
        Z = np.random.normal(size=simulation_matrix.shape)
        payoffs_t = self.calculate_digital_payoff(simulation_matrix[-1, :])
        delta = e**(-self.r * self.T) * payoffs_t * Z[-1, :]/(self.sigma*self.S*sqrt(self.T))
        return delta

# Parameters Exercise 1
S = 100
K = 99
T = 1.0
r = 0.06
sigma = 0.2
style = "european"
option_type = "put"


# Create option object
option = Option(S, K, T, r, sigma, style=style, type=option_type)

# Perform Simulations and Convergence Test
print(f"------------ Monte Carlo Simulations ------------")
convergence = []
ci = []
x = np.logspace(1,6, 6)
for i in x:
    simulation_matrix = option.monte_carlo_simulation(1, i, viz=False, seed=seed  )
    mc_option, std_error = option.monte_carlo_option_value(simulation_matrix)
    convergence.append(mc_option)
    ci.append(1.96 * std_error)
    print(f"Simulations: {i} | Mean Option Value: {mc_option:.4f} | Standard Error: {std_error:.4f} | "
          f"95% C.I.: [{mc_option - 1.96 * std_error:.4f},{mc_option + 1.96 * std_error:.4f}]")

# Display Convergence Test
plt.plot(x, convergence, label="Monte Carlo")
plt.fill_between(x, (np.array(convergence) - np.array(ci)), (np.array(convergence) + np.array(ci)), color='blue', alpha=0.1)
plt.hlines(option.blackscholes_option_value(), 0, x.max(), linestyles="dashed", colors='orange', label="Black-Scholes")
plt.xscale('log')
plt.legend(loc="upper right")
plt.xlabel("Number of Simulations")
plt.ylabel("Option Price")
plt.title("Convergence test")
plt.show()

# Simulate
i = 10**5
# Varying Strike Price
print(f"------------ Option Prices varying with K ------------")
K_prices = np.arange(80, 122, 2)
K_study = []
K_study_lowsim = []
K_study_bs = []
ci = []
ci_lowsim = []
for k in K_prices:
    # Create option object
    option = Option(S, k, T, r, sigma, style=style, type=option_type)
    simulation_matrix = option.monte_carlo_simulation(1, i, viz=False, seed=seed)
    mc_option, std_error = option.monte_carlo_option_value(simulation_matrix)
    K_study.append(mc_option)
    ci.append(1.96 * std_error)
    K_study_bs.append(option.blackscholes_option_value())
    print(f"K: {k} | Simulations: {i} | Mean Option Value: {mc_option:.2f} | Standard Error: {std_error:.2f} | "
          f"95% C.I.: [{mc_option - 1.96 * std_error:.2f},{mc_option + 1.96 * std_error:.2f}]")
    # For 10 simulations
    mc_option_lowsim, std_error_lowsim = option.monte_carlo_option_value(option.monte_carlo_simulation(1, 10, viz=False, seed=seed))
    K_study_lowsim.append(mc_option_lowsim)
    ci_lowsim.append(1.96 * std_error_lowsim)

# Display option price variation with K
plt.plot(K_prices, K_study, label=f"MC (10^5)", linewidth=3)
plt.fill_between(K_prices, (np.array(K_study) - np.array(ci)), (np.array(K_study) + np.array(ci)), color='blue', alpha=0.1)
plt.plot(K_prices, K_study_lowsim, label=f"MC (10^1)", color="tomato")
plt.fill_between(K_prices, (np.array(K_study_lowsim) - np.array(ci_lowsim)),
                 (np.array(K_study_lowsim) + np.array(ci_lowsim)), color='tomato', alpha=0.1)
plt.plot(K_prices, K_study_bs, label=f"Black-Scholes", color="orange")
plt.legend(loc="upper left")
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.title(f"Option ({option.type}) Price with respect to K")
plt.show()

# Varying Volatility
print(f"------------ Option Prices varying with Sigma ------------")
sigma_values = np.arange(0.01, 0.60, 0.01)
sigma_study = []
sigma_study_lowsim = []
sigma_study_bs = []
ci = []
ci_lowsim = []
for o in sigma_values:
    # Create option object
    option = Option(S, K, T, r, o, style=style, type=option_type)
    simulation_matrix = option.monte_carlo_simulation(1, i, viz=False, seed=seed)
    mc_option, std_error = option.monte_carlo_option_value(simulation_matrix)
    sigma_study.append(mc_option)
    ci.append(1.96 * std_error)
    sigma_study_bs.append(option.blackscholes_option_value())
    print(f"Sigma: {o:.2f} | Simulations: 2^({i}) | Mean Option Value: {mc_option:.2f} | Standard Error: {std_error:.2f} | "
          f"95% C.I.: [{mc_option - 1.96 * std_error:.2f},{mc_option + 1.96 * std_error:.2f}]")
    # For 2^10 simulations
    mc_option_lowsim, std_error_lowsim = option.monte_carlo_option_value(
        option.monte_carlo_simulation(1, 10, viz=False, seed=seed))
    sigma_study_lowsim.append(mc_option_lowsim)
    ci_lowsim.append(1.96 * std_error_lowsim)

# Display option price variation with volatility (sigma)
plt.plot(sigma_values, sigma_study, label=f"MC (10^5)", linewidth=3)
plt.fill_between(sigma_values, (np.array(sigma_study) - np.array(ci)), (np.array(sigma_study) + np.array(ci)), color='blue', alpha=0.1)
plt.plot(sigma_values, sigma_study_lowsim, label=f"MC (10^1)", color="tomato")
plt.fill_between(sigma_values, (np.array(sigma_study_lowsim) - np.array(ci_lowsim)),
                 (np.array(sigma_study_lowsim) + np.array(ci_lowsim)), color='tomato', alpha=0.1)
plt.plot(sigma_values, sigma_study_bs, label=f"Black-Scholes", color="orange")
plt.legend(loc="upper left")
plt.xlabel("Volatility")
plt.ylabel("Option Price")
plt.title(f"Option ({option.type}) Price with respect to Volatility")
plt.show()



# Hedging Parameter - Bump and Revalue
print(f"------------ Bump and Revalue ------------")
epsilons = np.array([0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.03, 0.05, 0.1])*option.S
x = np.logspace(1, 5, 10, endpoint=True)
eps_deltas = []
eps_std_errors = []
eps_relative_errors = []
for epsilon in epsilons:
    deltas = []
    std_errors = []
    relative_errors = []
    for i in x:
        simulation_matrix = option.monte_carlo_simulation(52, i, viz=False, seed=seed)
        delta, std_error = option.bump_and_revalue_delta(simulation_matrix, epsilon=epsilon)
        relative_error = abs((delta - option.bs_delta()))/abs(option.bs_delta())
        deltas.append(delta)
        std_errors.append(std_error)
        relative_errors.append(relative_error)
        print(f"Epsilon: {epsilon} | Simulations: {i} | Delta Value: {delta:.3f} | Relative Error: {relative_error*100:.3f}%")
    eps_deltas.append(deltas)
    eps_std_errors.append(std_errors)
    eps_relative_errors.append(relative_errors)

# Display Convergence Test for Each Epsilon
for a in range(len(epsilons)):
    plt.plot(x, np.array(eps_relative_errors[a])*100, label=f"epsilon = {epsilons[a]:.3f}")
    lb, ub = np.array(eps_deltas[a])- 1.96*np.array(eps_std_errors[a]),\
             np.array(eps_deltas[a])+ 1.96*np.array(eps_std_errors[a])
    #plt.fill_between(x, lb, ub, alpha=0.1)
plt.xscale('log')
plt.legend(loc="upper right")
plt.xlabel("Number of Simulations")
plt.ylabel("Delta Value (Absolute Relative Error %)")
plt.title("Bump and Revalue Method")
plt.show()

# Display Epsilons and Delta
simulation_matrix = option.monte_carlo_simulation(52, 10*5, viz=False, seed=seed)
epsilons = np.linspace(0.0000000001, 10, 100000)
g = []
for a in epsilons:
    delta, std_error = option.bump_and_revalue_delta(simulation_matrix, epsilon=a)
    relative_error = abs((delta - option.bs_delta())) / abs(option.bs_delta())
    g.append(relative_error)
plt.plot(epsilons, g)
plt.xlabel("Epsilons")
plt.ylabel("Delta Value (Relative Error %)")
plt.title("Bump and Revalue Method")
plt.show()


# Change parameter option_type
option_type = "call"
# Create option object
option = Option(S, K, T, r, sigma, style=style, type=option_type)
# Simulate
steps = 1
i = 10**6
# Hedging Parameter - Pathwise Method
print(f"------------ Pathwise Method ------------")
simulation_matrix = option.monte_carlo_simulation(steps, i, viz=False, seed=seed)
delta = option.pathwise_method(simulation_matrix, 0.5).mean()
print(f"Simulations: {i} | Delta Value: {delta:.6f}")
std = np.linspace(0.1, 25, 100)
oiois = []
for s in std:
    oioi = option.pathwise_method(simulation_matrix, s).mean()
    oiois.append(oioi)
plt.plot(std, oiois)
plt.title("Pathwise Method and the Smoothing of the Step Function")
plt.xlabel("Standard Deviation")
plt.ylabel("Delta")
plt.show()

# Hedging Parameter - Likelihood Ratio
print(f"------------ Likelihood Ratio ------------")
simulation_matrix = option.monte_carlo_simulation(steps, i, viz=False, seed=seed)
delta = option.likelihood_ratio_method(simulation_matrix).mean()
print(f"Simulations: {i} | Delta Value: {delta:.6f}")
# Hedging Parameter - Bump And Revalue
print(f"------------ Bump And Revalue ------------")
simulation_matrix = option.monte_carlo_simulation(steps, i, viz=False, seed=seed)
delta, std_error = option.bump_and_revalue_delta_digital(simulation_matrix, epsilon=0.0001*option.S)
print(f"Simulations: {i} | Delta Value: {delta:.6f}")
# Hedging Parameter - Black-Scholes
print(f"------------ Black-Scholes Analytical Value ------------")
print(f"Delta Value: {option.bs_delta():.6f}")


# Plot Pathwise
sims = np.logspace(1,5, 20)
likelihoods = []
pathwisesss = []
bumpieeesss = []
for sim in sims:
    simulation_matrix = option.monte_carlo_simulation(1, sim, viz=False, seed=seed)
    # Likelihood Method
    like_met = option.likelihood_ratio_method(simulation_matrix).mean()
    path_met = option.pathwise_method(simulation_matrix).mean()
    bump_met, std_error = option.bump_and_revalue_delta_digital(simulation_matrix, epsilon=0.001 * option.S)
    likelihoods.append(like_met)
    pathwisesss.append(path_met)
    bumpieeesss.append(bump_met)
plt.plot(sims, likelihoods, label="Likelihood Method")
plt.plot(sims, pathwisesss, label="Pathwise Method")
plt.plot(sims, bumpieeesss, label="Bump and Revalue Method")
plt.xscale('log')
plt.legend(loc="upper right")
plt.title("Hedge Parameter")
plt.xlabel("Standard Deviation")
plt.ylabel("Delta")
plt.show()
