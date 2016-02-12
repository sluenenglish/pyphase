import unittest
import pyphase
import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import integers, floats, lists, composite

class TestModel(unittest.TestCase):

    @composite
    def valid_phase_type_generator(draw, k):
        ptg = []
        for i in range(k):
            ptg.append(draw(lists(floats(min_value=0, max_value=1),
                                                min_size=k, max_size=k)))
        ptg_array = np.array(ptg)
        ptg_totals = np.array(draw(lists(floats(min_value = 0, max_value=1),
                                                min_size=k, max_size=k))).reshape(-1, 1)
        with np.errstate(invalid='ignore'):
            ptg_norm  = (np.nan_to_num(ptg_array / ptg_array.sum(axis=1).reshape(-1,1))) \
                                * ptg_totals
        return ptg_norm


    @composite
    def valid_initial_vector(draw, k):
        initial_vector = np.array(draw(lists(floats(min_value=0, max_value=1),
                                                min_size=k, max_size=k)))
        assume(initial_vector.sum() > 0)
        initial_vector_norm = initial_vector / initial_vector.sum()
        return initial_vector_norm

