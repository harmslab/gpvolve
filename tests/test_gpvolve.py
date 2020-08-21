#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `gpvolve` package."""

import pytest
from gpvolve import GenotypePhenotypeMSM
from gpmap import GenotypePhenotypeMap
from gpgraph import GenotypePhenotypeGraph
import numpy as np


@pytest.fixture()
def gpvolve_base():
    wildtype = "AAA"

    genotypes = [
        "AAA",
        "AAB",
        "ABA",
        "BAA",
        "ABB",
        "BAB",
        "BBA",
        "BBB"
    ]

    binary = [
        '000',
        '001',
        '010',
        '100',
        '011',
        '101',
        '110',
        '111'
    ]

    mutations = {
        0: ["A", "B"],
        1: ["A", "B"],
        2: ["A", "B"],
    }
    phenotypes = np.random.rand(len(genotypes))
    tmp_gpm_file = GenotypePhenotypeMap(wildtype=wildtype,
                                        genotypes=genotypes,
                                        phenotypes=phenotypes,
                                        log_transform=False,
                                        mutations=mutations)
    gpv = GenotypePhenotypeMSM(tmp_gpm_file)
    return gpv


def test_gpvolve(gpvolve_base):
    """Test type of object"""
    assert isinstance(gpvolve_base, GenotypePhenotypeMSM)


def test_inner_gpmap(gpvolve_base):
    assert isinstance(gpvolve_base.gpm, GenotypePhenotypeMap)


# def test_attributes(gpvolve_base):
#     assert isinstance()
