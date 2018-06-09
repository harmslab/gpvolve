#!/Users/leandergoldbach/miniconda3/bin/python

import sys
import json
import matplotlib.pyplot as plt
import numpy as np

class MonteCarloAnalysis(object):
    def __init__(self, jsonfile):

        # Read data and path IDs from .json file.
        self.data, self.paths = self.read_log(jsonfile)

        # Set filename
        self.filename = (jsonfile.split("/")[-1]).split(".")[0]

    def read_json(self, jsonfile):
        # Open, json load, and close a json file
        f = open(jsonfile, "r")
        data = json.load(f)
        f.close()
        return data

    def read_log(self, jsonfile):
        """ Read .json, and return data and path IDs"""
        masterlog = self.read_json(jsonfile)
        paths = []
        for path in masterlog:
            try:
                int(path)
                paths.append(tuple(masterlog[path]["path"]))
            except ValueError:
                pass

        return masterlog, paths

    def histogram(self):
        numb_of_paths = len(self.paths)

        # Normalize counts by total number of counts.
        norm_counts = []
        for val in self.count():
            norm_counts.append(val / numb_of_paths)

        x = range(1, len(norm_counts)+1)
        y = norm_counts

        fig, ax = plt.subplots()
        plt.xlabel("Path #")
        plt.ylabel("Fraction (Total # of Paths: %s)" % numb_of_paths)
        ax.bar(x, y)

        plt.savefig("%s_histo.pdf" % self.filename, format='pdf', dpi=300)

    def count(self):
        counts = []
        for path in self.unique_sorted_paths():
            counts.append(self.paths.count(path))
        return tuple(counts)


    def sort_paths_by_id(self):
        """Sort path  tuples by first to last position. """
        # Example: [(0, 2, 13, 24), (0, 1, 28, 30), (0, 2, 15, 22)] -> [(0, 1, 28, 30), (0, 2, 13, 24), (0, 2, 15, 22)].
        # Lamdba is an anonymous function that tells sorted() to take tup as input and then apply sorted on tup[:].
        # i.e. sort by all elements in tup. key=lambda tup: tup[1] would only sort on the second element in the tuple.
        sorted_pathlist = sorted(self.paths, key=lambda tup: tup[:])

        return sorted_pathlist

    def unique_sorted_paths(self):
        """ Return unique sorted list of path IDs """
        uniq = list(set(self.paths))
        uniq_sorted_pathlist = sorted(uniq, key=lambda tup: tup[:])
        c = 0
        for i in uniq_sorted_pathlist:
            c += 1
            print("Path %s: %s, length: %s" % (c, i, len(i)))

        return uniq_sorted_pathlist


path_sample = MonteCarloAnalysis(sys.argv[1])
print("Number of unique Paths: %s" % len(path_sample.unique_sorted_paths()))
path_sample.histogram()
# print(path_sample.unique_sorted_paths())