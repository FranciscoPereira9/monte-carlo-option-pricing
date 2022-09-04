# monte-carlo-option-pricing

The code in this repository was developed for the Assignment 2 of Computational Finance at UvA.

This repository includes:
1. A computer program used to price European Options with the Monte Carlo method. The following is discussed:
	- Convergence studies, achieved by increasing the number of trials
	- Numerical tests for varying values of the strike and the volatility parameter.
	- Standard error and accuracy

2. Hedging, using the following methods:
	- Bump and Revalue
	- Pathwise
	- Likelihood Ratio

3. Variance reduction techniques, for Monte Carlo simulations, in Asian options:
	- Pricing of Asian options (arithmetic vs geometric averages)
	- Control Variate Strategy
	- Performance study for different parameter settings (number of paths, strike, number of time points used in the average, etc.)
