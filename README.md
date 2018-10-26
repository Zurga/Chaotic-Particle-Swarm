# Chaotic Particle Swarm optimization

A Python implementation of the article: "Improved particle swarm optimization combined with chaos" by Bo Liua, Ling Wang, Yi-Hui Jin Fang Tang, De-Xian Huang. https://www.sciencedirect.com/science/article/pii/S0960077905000330

## Abtract from the paper:
As a novel optimization technique, chaos has gained much attention and some applications during the past decade. For a given energy or cost function, by following chaotic ergodic orbits, a chaotic dynamic system may eventually reach the global optimum or its good approximation with high probability. To enhance the performance of particle swarm optimization (PSO), which is an evolutionary computation technique through individual improvement plus population cooperation and competition, hybrid particle swarm optimization algorithm is proposed by incorporating chaos. Firstly, adaptive inertia weight factor (AIWF) is introduced in PSO to efficiently balance the exploration and exploitation abilities. Secondly, PSO with AIWF and chaos are hybridized to form a chaotic PSO (CPSO), which reasonably combines the population-based evolutionary searching ability of PSO and chaotic searching behavior. Simulation results and comparisons with the standard PSO and several meta-heuristics show that the CPSO can effectively enhance the searching efficiency and greatly improve the searching quality.

## Installation
<code>
pip install -r requirements.txt

python setup.py install
<code>

## Example usage:
<code>
from chaotic_particle_swarm import ChaosSwarm

function = lambda x: x ** 2

search_space = [(-100, 100)]

optimal = ChaosSwarm(function, search_space, num_particles=50).run()

print('Optimal values are:', optimal)

print('Result of function with optimal values:', function(*optimal))
</code>
