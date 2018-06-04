#!/Users/leandergoldbach/miniconda3/bin/python

from __future__ import division
from math import e
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_discrete






xk = np.arange(7)
pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)

samples = []
for i in range(0,100000,1):
    custm = rv_discrete(name='custm', values=(xk, pk)).rvs()
    samples.append(custm)

uniqlist = sorted(list(set(samples)))

counts = []
for sample in uniqlist:
    counts.append(samples.count(sample)/len(samples))

print(counts)

fig, ax = plt.subplots(1, 1)
ax.scatter(uniqlist, counts,)

# ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
# ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)


#
# def fixation_probability(rel_current, rel_proposed):
#     #fix_prob = (1 - (e ** (-fratio))) / (1 - (e ** (-10)
#     #fix_prob = (1 - e ** -(1 - (1/fratio))) / (1 - e ** -(100000 * (1 - (1/fratio)))) # fratio = rel_current / rel_proposed
#     #fix_prob = (1 - e ** -(1 - (rel_current / rel_proposed))) / (1 - e ** -(10000 * (1 - (rel_current / rel_proposed))))
#     fix_prob = 1 - e ** -((rel_proposed / rel_current)-1) / 1 - e ** -10000 * ((rel_proposed / rel_current)-1)
#     #fix_prob = 1 - math.e ** (-1 * fratio) / 1 - math.e ** (-1 * pop_size * (fratio))
#
#     return fix_prob
#
#
# fig, ax, = plt.subplots()
#
# y = []
# x = []
#
# rel_current = 0.5
#
# # fixation_probability(0.99, 1.0)
#
#
# # for exponent in range(1,2):
# #     population_size = 10000 ** exponent
# for i in np.arange(0.1,5,0.1):
#     x.append(i/rel_current)
#     y.append(fixation_probability(rel_current, i))
# ax.plot(x, y, linestyle=":", alpha = 0.3)
#
#
# plt.axvline(x=1, color='black', alpha=0.5, linestyle=':', linewidth= 1)
# ax.set_ylabel("Fixation probability")
# ax.set_xlabel("Fitness ratio")
# # ax.text(0.0, 0.0, 'Population size: %s' % population_size, ha='right', fontsize=8)



plt.savefig("draw_from_discr_distr.pdf", format='pdf', dpi=300)

# fit_curr = 0.00005
# fit_prop = 0.00001
# print("fcurr: %s, fixprop: %s" % (fit_curr, fit_prop), "\n",  fixation_probability(fit_curr, fit_prop, 1000000))