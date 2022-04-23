from __future__ import division
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


# This class was found here https://chart-studio.plotly.com/~alanleethompson/36/a-python-class-to-price-asian-options-wi/#/
# The GeometricAsianOption2 property was adapted to be coherent with the Assignment2 formula for the analytical value
class AsianOptionMC_MC(object):
    """ Class for Asian options pricing using control variate
    S0 : float : initial stock/index level
    strike : float : strike price
    T : float : time to maturity (in year fractions)
    M : int : grid or granularity for time (in number of total points)
    r : float : constant risk-free short rate
    div :    float : dividend yield
    sigma :  float : volatility factor in diffusion term

    Unitest (doctest):
    >>> myAsianCall = AsianOptionMC_MC('call', 4., 4., 1., 100., .03, 0, .25, 10000)
    >>> myAsianCall.value
    (0.26141622329842962, 0.25359138249114327, 0.26924106410571597)
    >>> myAsianCall.value_with_control_variate
    (0.25813771282805958, 0.25771678775128265, 0.25855863790483652)

    """

    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, simulations):
        try:
            self.option_type = option_type
            assert isinstance(option_type, str)
            self.S0 = float(S0)
            self.strike = float(strike)
            self.T = float(T)
            self.M = int(M)
            self.r = float(r)
            self.div = float(div)
            self.sigma = float(sigma)
            self.simulations = int(simulations)
        except ValueError:
            print('Error passing Options parameters')

        if option_type != 'call' and option_type != 'put':
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(- self.r * self.T)

    @property
    def GeometricAsianOption(self):
        sigsqT = ((self.sigma ** 2 * self.T * (self.M + 1) * (2 * self.M + 1))
                  / (6 * self.M * self.M))
        muT = (0.5 * sigsqT + (self.r - 0.5 * self.sigma ** 2)
               * self.T * (self.M + 1) / (2 * self.M))
        d1 = ((np.log(self.S0 / self.strike) + (muT + 0.5 * sigsqT))
              / np.sqrt(sigsqT))
        d2 = d1 - np.sqrt(sigsqT)
        geometric_value = self.discount * (self.S0 * np.exp(muT) * norm.cdf(d1) - self.strike * norm.cdf(d2))
        return geometric_value

    @property
    def GeometricAsianOption2(self):
        sig_tilt = self.sigma * np.sqrt((2 * self.M + 1) / (6 * (self.M + 1)))
        r_tilt = 0.5 * (self.r - 0.5 * self.sigma ** 2 + sig_tilt ** 2)
        d1_tilt = (np.log(self.S0 / self.strike) + (r_tilt + 0.5 * sig_tilt**2)*self.T) / (np.sqrt(T)*sig_tilt)
        d2_tilt = (np.log(self.S0 / self.strike) + (r_tilt - 0.5 * sig_tilt**2)*self.T) / (np.sqrt(T)*sig_tilt)
        geometric_value = self.discount * (self.S0 * np.exp(r_tilt*T) * norm.cdf(d1_tilt) - self.strike * norm.cdf(d2_tilt))
        return geometric_value

    @property
    def price_path(self, seed=100):
        np.random.seed(seed)
        price_path = (self.S0 *
                      np.cumprod(np.exp((self.r - 0.5 * self.sigma ** 2) * self.time_unit +
                                        self.sigma * np.sqrt(self.time_unit)
                                        * np.random.randn(self.simulations, self.M)), 1))
        return price_path

    @property
    def MCPayoff(self):
        if self.option_type == 'call':
            MCpayoff = self.discount \
                       * np.maximum(np.mean(self.price_path, 1) - self.strike, 0)
        else:
            MCpayoff = self.discount \
                       * np.maximum(self.strike - np.mean(self.price_path, 1), 0)
        return MCpayoff

    @property
    def value(self):
        MCvalue = np.mean(self.MCPayoff)
        MCValue_std = np.std(self.MCPayoff)
        upper_bound = MCvalue + 1.96 * MCValue_std / np.sqrt(self.simulations)
        lower_bound = MCvalue - 1.96 * MCValue_std / np.sqrt(self.simulations)
        return MCvalue, lower_bound, upper_bound

    @property
    def value_with_control_variate(self):
        geometric_average = np.exp((1 / float(self.M)) * np.sum(np.log(self.price_path), 1))
        if self.option_type == 'call':
            MCpayoff_geometric = self.discount * np.maximum(geometric_average - self.strike, 0)
        else:
            MCpayoff_geometric = self.discount * np.maximum(self.strike - geometric_average, 0)
        value_with_CV = self.MCPayoff + self.GeometricAsianOption - MCpayoff_geometric
        value_with_control_variate = np.mean(value_with_CV, 0)
        value_with_control_variate_std = np.std(value_with_CV, 0)
        upper_bound_CV = value_with_control_variate + 1.96 * value_with_control_variate_std / np.sqrt(self.simulations)
        lower_bound_CV = value_with_control_variate - 1.96 * value_with_control_variate_std / np.sqrt(self.simulations)
        return value_with_control_variate, lower_bound_CV, upper_bound_CV


S = 100
K = 99
T = 1
r = 0.06
sigma = 0.20
M = 52 #steps
n_runs = 1000 #simulations

myAsianCall = AsianOptionMC_MC('call', S, K, T, M, r, 0, sigma, n_runs)
print("Asian Option Value: ", myAsianCall.value)
print("Geometric Asian Option: ", myAsianCall.GeometricAsianOption2)
print("Control Variate: ", myAsianCall.value_with_control_variate)
#print("Geometric Asian Option: ", myAsianCall.GeometricAsianOption)


# Study impact of Number of Simulations
x = np.logspace(1, 6, 24, endpoint=True)
asian_option_values = []
asian_option_lb = []
asian_option_up = []
control_variate = []
control_variate_lb = []
control_variate_up = []
for i in x:
    myAsianCall = AsianOptionMC_MC('call', S, K, T, M, r, 0, sigma, i)
    # Monte Carlo Arithmetic Averages
    value, lb, up = myAsianCall.value
    asian_option_values.append(value)
    asian_option_lb.append(lb)
    asian_option_up.append(up)
    # Monte Carlo Control Variate
    value, lb, up = myAsianCall.value_with_control_variate
    control_variate.append(value)
    control_variate_lb.append(lb)
    control_variate_up.append(up)

# Display Convergence Test
plt.plot(x, asian_option_values, label="Monte Carlo: Arithmetic Avg", color="blue")
plt.fill_between(x, asian_option_lb, asian_option_up, color='blue', alpha=0.1)
plt.plot(x, control_variate, label="Monte Carlo: Control Variate", color="red")
plt.fill_between(x, control_variate_lb, control_variate_up, color='red', alpha=0.1)
plt.hlines(myAsianCall.GeometricAsianOption2, x.min(), x.max(), linestyles="dashed", colors='orange', label="Analytical: Geometric Avg")
plt.xscale('log')
plt.legend(loc="upper right")
plt.xlabel("Number of Simulations")
plt.ylabel("Asian Option Price")
plt.title("Convergence Test - Asian Option")
plt.show()


# Study Impact of Strike Price
x = np.linspace(80, 120, 40, endpoint=True)
asian_option_values = []
asian_option_lb = []
asian_option_up = []
control_variate = []
control_variate_lb = []
control_variate_up = []
analytical_values = []
for strike in x:
    myAsianCall = AsianOptionMC_MC('call', S, strike, T, M, r, 0, sigma, n_runs)
    # Monte Carlo Arithmetic Averages
    value, lb, up = myAsianCall.value
    asian_option_values.append(value)
    asian_option_lb.append(lb)
    asian_option_up.append(up)
    # Monte Carlo Control Variate
    value, lb, up = myAsianCall.value_with_control_variate
    control_variate.append(value)
    control_variate_lb.append(lb)
    control_variate_up.append(up)
    # Monte Carlo Analytical
    analytical_values.append(myAsianCall.GeometricAsianOption2)

# Display Strike Test
plt.plot(x, asian_option_values, label="Monte Carlo: Arithmetic Avg", color="blue")
plt.fill_between(x, asian_option_lb, asian_option_up, color='blue', alpha=0.1)
plt.plot(x, control_variate, label="Monte Carlo: Control Variate", color="red")
plt.fill_between(x, control_variate_lb, control_variate_up, color='red', alpha=0.1)
plt.plot(x, analytical_values, label="Analytical: Geometric Avg", linestyle="dashed", color="orange")
plt.legend(loc="upper right")
plt.xlabel("Strike Price")
plt.ylabel("Asian Option Price")
plt.title("Asian Option Price with respect to Strike Price")
plt.show()

# Study Impact of Volatility
x = np.linspace(0.01, 0.6, 30, endpoint=True)
asian_option_values = []
asian_option_lb = []
asian_option_up = []
control_variate = []
control_variate_lb = []
control_variate_up = []
analytical_values = []
for vol in x:
    myAsianCall = AsianOptionMC_MC('call', S, K, T, M, r, 0, vol, n_runs)
    # Monte Carlo Arithmetic Averages
    value, lb, up = myAsianCall.value
    asian_option_values.append(value)
    asian_option_lb.append(lb)
    asian_option_up.append(up)
    # Monte Carlo Control Variate
    value, lb, up = myAsianCall.value_with_control_variate
    control_variate.append(value)
    control_variate_lb.append(lb)
    control_variate_up.append(up)
    # Monte Carlo Analytical
    analytical_values.append(myAsianCall.GeometricAsianOption2)

# Display Strike Test
plt.plot(x, asian_option_values, label="Monte Carlo: Arithmetic Avg", color="blue")
plt.fill_between(x, asian_option_lb, asian_option_up, color='blue', alpha=0.1)
plt.plot(x, control_variate, label="Monte Carlo: Control Variate", color="red")
plt.fill_between(x, control_variate_lb, control_variate_up, color='red', alpha=0.1)
plt.plot(x, analytical_values, label="Analytical: Geometric Avg", linestyle="dashed", color="orange")
plt.legend(loc="upper left")
plt.xlabel("Volatility")
plt.ylabel("Asian Option Price")
plt.title("Asian Option Price with respect to Volatility")
plt.show()


# Study Impact of Number of Steps
x = np.linspace(1, 365, 100, endpoint=True)
asian_option_values = []
asian_option_lb = []
asian_option_up = []
control_variate = []
control_variate_lb = []
control_variate_up = []
analytical_values = []
for steps in x:
    myAsianCall = AsianOptionMC_MC('call', S, K, T, steps, r, 0, sigma, n_runs)
    # Monte Carlo Arithmetic Averages
    value, lb, up = myAsianCall.value
    asian_option_values.append(value)
    asian_option_lb.append(lb)
    asian_option_up.append(up)
    # Monte Carlo Control Variate
    value, lb, up = myAsianCall.value_with_control_variate
    control_variate.append(value)
    control_variate_lb.append(lb)
    control_variate_up.append(up)
    # Monte Carlo Analytical
    analytical_values.append(myAsianCall.GeometricAsianOption2)

# Display Strike Test
plt.plot(x, asian_option_values, label="Monte Carlo: Arithmetic Avg", color="blue")
plt.fill_between(x, asian_option_lb, asian_option_up, color='blue', alpha=0.1)
plt.plot(x, control_variate, label="Monte Carlo: Control Variate", color="red")
plt.fill_between(x, control_variate_lb, control_variate_up, color='red', alpha=0.1)
plt.plot(x, analytical_values, label="Analytical: Geometric Avg", linestyle="dashed", color="orange")
plt.legend(loc="upper right")
plt.xlabel("Number of Steps")
plt.ylabel("Asian Option Price")
plt.title("Asian Option Price with respect to Number of Steps")
plt.show()