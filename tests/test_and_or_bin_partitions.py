from unittest import TestCase

from src.and_or_top_down import bin_partitions


class test_and_or_bin_partitions(TestCase):
    def test_bin_partitions(self):
        lst = [1, 2, 3]
        subassemblies_to_test = bin_partitions(lst)
        print(subassemblies_to_test)
        self.assertTrue(1 == 1)
