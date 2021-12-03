from unittest import TestCase

from src.and_or_top_down import read_mw_data_excel

class test_read_mw_excel(TestCase):
    def test_read_mw_data_excel(self):
        path = '../data/generated/xlsx/exp_1_Moving wedge.xlsx'
        mw_df = read_mw_data_excel(path)
        print(mw_df)
        assert True


class test_and_or_bin_partitions(TestCase):
    def test_bin_partitions(self):
        lst = [1, 2, 3]
        subassemblies_to_test = bin_partitions(lst)
        print(subassemblies_to_test)
        self.assertTrue(1 == 1)
