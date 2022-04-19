# import os
# import sys
#
# PROJECT_PATH = os.getcwd()
# SOURCE_PATH = os.path.join(PROJECT_PATH, 'src')
# sys.path.append(SOURCE_PATH)
#
import numpy as np
import json
from dash_data_viewer.dash_hdf import HdfId, DashHDF, HDFFileHandler
import unittest


class Test(unittest.TestCase):
    @staticmethod
    def get_hdf(name: str, mode, number: int = 1) -> DashHDF:
        """Note: Relies on get_id and get_hdf"""
        hdf_id = HdfId(page='test', experiment=name, number=number)
        hdf = DashHDF(hdf_id, mode=mode)
        return hdf

    def test_get_id(self):
        """Get the id of an HDF file"""
        hdf_id = HdfId(page='test', experiment='testexp', number=1)
        self.assertEqual('test-testexp-1', hdf_id.uid)

    def test_get_hdf(self):
        """Get an hdf file"""
        hdf_id = HdfId(page='test', experiment='testexp', number=1)
        hdf = DashHDF(hdf_id)
        self.assertEqual(hdf_id, hdf.hdf_id)

    def test_read_data(self):
        """Read array from HDF file"""
        hdf = self.get_hdf('read', mode='r')
        expected = np.array([1, 2, 3, 4])
        with HDFFileHandler(hdf.hdf_path, 'w') as f:
            f['test_read_data'] = expected
        with hdf:
            read = hdf.get_data('test_read_data', subgroup=None)

        self.assertTrue(np.all(expected == read))

    def test_write_data(self):
        hdf = self.get_hdf('write', mode='w')
        expected = np.array([1, 2, 3, 4])
        with hdf:
            hdf.save_data(expected, name='test_write_data', subgroup=None)

        hdf.mode = 'r'
        with hdf:
            data = hdf.get_data('test_write_data', subgroup=None)
        self.assertTrue(np.all(expected == data))

    def test_read_info(self):
        hdf = self.get_hdf('read', mode='r')
        expected = dict(a='a', one=1)
        with HDFFileHandler(hdf.hdf_path, 'w') as f:
            f.attrs['test_read_info'] = json.dumps(expected)  # TODO: Might want to replace this with whatever function does the usual writing

        with hdf:
            info = hdf.get_info(name='test_read_info', subgroup=None)
        self.assertEqual(expected, info)

    def test_write_info(self):
        hdf = self.get_hdf('write', mode='w')
        expected = dict(a='a', one=1)
        with hdf:
            hdf.save_info(expected, name='test_write_data', subgroup=None)

        hdf.mode = 'r'
        with hdf:
            info = hdf.get_info('test_write_data', subgroup=None)
        self.assertEqual(expected, info)
