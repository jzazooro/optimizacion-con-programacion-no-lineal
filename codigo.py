import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
np.random.seed(0)

TOTAL_BUDGET = 100_000

# Alpha and beta constants
alphas = np.array([-3300, -1233, -2633])
betas = np.array([4533, 5833, 3333])

# Linearly spaced numbers
x = np.linspace(1, TOTAL_BUDGET, TOTAL_BUDGET)

# Variables
hamburguesa   = cp.Variable(pos=True)
patatas = cp.Variable(pos=True)
croissants  = cp.Variable(pos=True)

# Constraint
constraint = [hamburguesa + patatas + croissants <= TOTAL_BUDGET]

# Objective
obj = cp.Maximize(alphas[0] + betas[0] * cp.log(hamburguesa)
                + alphas[1] + betas[1] * cp.log(patatas)
                + alphas[2] + betas[2] * cp.log(croissants))

# Solve
prob = cp.Problem(obj, constraint)
prob.solve(solver='ECOS', verbose=False)

# Print solution
print('='*59 + '\n' + ' '*24 + 'Solution' + ' '*24 + '\n' + '='*59)
print(f'Status = {prob.status}')
print(f'Returns = ${round(prob.value):,}\n')
print('Hamburguesería Marketing allocation:')
print(f' - Anuncios de Hamburguesa   = ${round(hamburguesa.value):,}')
print(f' - Anuncios de Patatas = ${round(patatas.value):,}')
print(f' - Anuncios de Croissants  = ${round(croissants.value):,}')

# Plot the functions and the results
fig = plt.figure(figsize=(10, 5), dpi=300)
plt.plot(x, alphas[0] + betas[0] * np.log(x), color='red', label='Anuncios de Hamburguesa')
plt.plot(x, alphas[1] + betas[1] * np.log(x), color='blue', label='Anuncios de Patatas')
plt.plot(x, alphas[2] + betas[2] * np.log(x), color='green', label='Anuncios de Croissants')

# Plot optimal points
plt.scatter([hamburguesa.value, patatas.value, croissants.value],
            [alphas[0] + betas[0] * np.log(hamburguesa.value),
             alphas[1] + betas[1] * np.log(patatas.value),
             alphas[2] + betas[2] * np.log(croissants.value)],
             marker="+", color='black', zorder=10)

plt.xlabel('Budget ($)')
plt.ylabel('Returns ($)')
plt.legend()
plt.show()