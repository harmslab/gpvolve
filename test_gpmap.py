#!/Users/leandergoldbach/miniconda3/bin/python

import matplotlib.pyplot as plt
import numpy as np
import json
import re

from gpmap.utils import encode_mutations, construct_genotypes, DNA
from gpmap import GenotypePhenotypeMap

from gpmap.simulate import MountFujiSimulation
from gpgraph import GenotypePhenotypeGraph, draw_flattened
import networkx as nx
from networkx import google_matrix
from gpgraph.base import get_neighbors

gpmapfile = "/Users/leandergoldbach/Utrecht/Minor/evobiophys/genotype-phenotype-maps/weinreich.json"
filename = gpmapfile.split("/")[-1].split(".")[-2]

'''
# Wildtype sequence
wt = "AAA"

# Micro-managing here, stating explicitly what substitutions are possible at each site.
# See documentation for more detail.
mutations = {
    0:DNA,
    1:DNA,
    2:["A","T"]
}

# Generate encoding dictionary for each mutation
encoding = encode_mutations(wt, mutations)
#print(encoding)

# Construct genotypes from every possible combination of the substitutions above.
genotypes, binary = construct_genotypes(encoding)
#print(genotypes, binary)

# Generate random phenotype values
phenotypes = np.random.rand(len(genotypes))

#print(wt)
'''

with open(gpmapfile, "r") as json_file:
    gpm = json.load(json_file)

genotypes = gpm["genotypes"]
phenotypes = gpm["phenotypes"]
wt = gpm["wildtype"]
mutations = gpm["mutations"]


gpm = GenotypePhenotypeMap(wt, # wildtype sequence
                   genotypes, # genotypes
                   phenotypes, # phenotypes
                   stdeviations=None, # errors in measured phenotypes
                   log_transform=False, # Should the map log_transform the space?
                   mutations=mutations # Substitution map to alphabet
                           )

#print(gpm.data)

gt_index = gpm.data.index[gpm.data['genotypes'] == '00001']
# print(gt_index)
# print("XXXXXX")
# print(gpm.data.genotypes[gt_index])


G = GenotypePhenotypeGraph(gpm)
for genotype in gpm.genotypes:
    print(get_neighbors(genotype, gpm.mutations))

network = draw_flattened(G, with_labes=True, node_size=100, font_size=7)

plt.savefig("%s.pdf" % filename, format='pdf', dpi=300)

TM = google_matrix(G)

nodeslist = list(G.nodes.data())
#print(nodeslist)
# print("TM dimension, shape, size:", TM.ndim, TM.shape, TM.size)

# print(TM[0,:])
# print(TM[:,0])
