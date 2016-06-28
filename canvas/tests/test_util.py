from __future__ import division
from canvas.util import check_table_grouping, match
import numpy as np
import pandas as pd
from skbio.util import assert_data_frame_almost_equal
from skbio.util._testing import assert_series_almost_equal
import unittest


class UtilTests(unittest.TestCase):
    def setUp(self):
        # Basic count data with 2 groupings
        self.table1 = pd.DataFrame([
            [10, 10, 10, 20, 20, 20],
            [11, 12, 11, 21, 21, 21],
            [10, 11, 10, 10, 11, 10],
            [10, 11, 10, 10, 10, 9],
            [10, 11, 10, 10, 10, 10],
            [10, 11, 10, 10, 10, 11],
            [10, 13, 10, 10, 10, 12]]).T
        self.cats1 = pd.Series([0, 0, 0, 1, 1, 1])

        # zero count
        self.bad1 = pd.DataFrame(np.array([
            [10, 10, 10, 20, 20, 0],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10]]).T)
        # negative count
        self.bad2 = pd.DataFrame(np.array([
            [10, 10, 10, 20, 20, 1],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, -1],
            [10, 10, 10, 10, 10, 10]]).T)

        # missing count
        self.bad3 = pd.DataFrame(np.array([
            [10, 10, 10, 20, 20, 1],
            [11, 11, 11, 21, 21, 21],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, np.nan],
            [10, 10, 10, 10, 10, 10]]).T)
        self.badcats1 = pd.Series([0, 0, 0, 1, np.nan, 1])
        self.badcats2 = pd.Series([0, 0, 0, 0, 0, 0])
        self.badcats3 = pd.Series([0, 0, 1, 1])
        self.badcats4 = pd.Series(range(len(self.table1)))
        self.badcats5 = pd.Series([1]*len(self.table1))

    def test_fail_missing(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats1)

    def test_fail_groups(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats2)

    def test_fail_size_mismatch(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats3)

    def test_fail_group_unique(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats4)

    def test_fail_1_group(self):
        with self.assertRaises(ValueError):
            check_table_grouping(self.table1, self.badcats5)

    def test_match(self):
        table = pd.DataFrame([
            [10, 10, 10, 20, 20],
            [11, 12, 11, 21, 21],
            [10, 11, 10, 10, 10],
            [10, 11, 10, 10, 9],
            [10, 11, 10, 10, 10],
            [10, 11, 10, 10, 11],
            [10, 13, 10, 10, 12]]).T
        table.index = ['a', 'b', 'c', 'd', 'e']
        grouping = pd.Series([0, 1, 0, 1, 1],
                             index=['b', 'c', 'd', 'e', 'f'])

        res_table, res_grouping = match(table, grouping)
        exp_table = pd.DataFrame([
            [10, 10, 20, 20],
            [12, 11, 21, 21],
            [11, 10, 10, 10],
            [11, 10, 10, 9],
            [11, 10, 10, 10],
            [11, 10, 10, 11],
            [13, 10, 10, 12]],
            columns=['b', 'c', 'd', 'e']).T
        exp_grouping = pd.Series([0, 1, 0, 1],
                                 index=['b', 'c', 'd', 'e'])
        res_table = res_table.reindex(index=['b', 'c', 'd', 'e'])
        res_grouping = res_grouping.reindex(index=['b', 'c', 'd', 'e'])
        assert_data_frame_almost_equal(exp_table, res_table)
        assert_series_almost_equal(exp_grouping, res_grouping)

if __name__ == '__main__':
    unittest.main()
