from scipy.special import binom
from itertools import combinations
import numpy as np
from random import choices

class MetricSpaceTest():

    def __init__(self, testset = [], short = False):
        self.testset = testset
        self.short = short

    def test(self):
        print('\n', self.non_identity(self.testset))
        print('\n', self.self_similarity(self.testset))
        print('\n', self.symmetry(self.testset))
        print('\n', self.triangle_inequality(self.testset, short = self.short))

    @staticmethod
    def non_identity(fingerprint_list):
        print('Checking non-identic --> S < 1 property!')
        combs = binom(len(fingerprint_list), 2)
        truth = True
        for idx, (fp1, fp2) in enumerate(combinations(fingerprint_list, 2)):
            print(str(round(idx / combs * 100, 3)) + ' %', end = '\r')
            if fp1.get_similarity(fp2) == 1:
                if not fp1.get_data() == fp2.get_data():
                    return False
        return True

    @staticmethod
    def triangle_inequality(fingerprint_list, short = True):
        print("Checking triangle inequality!")
        if short:
            print('Short test on 150 random samples.')
        test_list = choices(fingerprint_list, k=150) if short else fingerprint_list
        combs = binom(len(test_list), 3)
        truth = True
        for idx, (fp1, fp2, fp3) in enumerate(combinations(test_list, 3)):
            print(str(round(idx / combs * 100, 3)) + ' %', end = '\r')
            truth = fp1.get_similarity(fp2) +  fp2.get_similarity(fp3) <=  fp1.get_similarity(fp3) + 1
            if not truth:
                break
        return truth

    @staticmethod
    def self_similarity(fingerprint_list):
        print('Checking self-similarity property!')
        truth = True
        for fingerprint in fingerprint_list:
            truth = truth and fingerprint.get_similarity(fingerprint) == 1
        return truth

    @staticmethod
    def symmetry(fingerprint_list):
        print('Checking symmetry property!')
        combs = binom(len(fingerprint_list), 2)
        truth = True
        for idx, (fp1, fp2) in enumerate(combinations(fingerprint_list, 2)):
            print(str(round(idx / combs * 100, 3)) + ' %', end = '\r')
            truth = fp1.get_similarity(fp2) ==  fp2.get_similarity(fp1)
            if not truth:
                break
        return truth
