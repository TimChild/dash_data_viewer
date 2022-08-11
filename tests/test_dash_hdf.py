# import os
# import sys
#
# PROJECT_PATH = os.getcwd()
# SOURCE_PATH = os.path.join(PROJECT_PATH, 'src')
# sys.path.append(SOURCE_PATH)
#
import h5py
import numpy as np
import json
from dash_data_viewer.dash_hdf import HdfId, DashHDF, HDFFileHandler
import unittest


class Test(unittest.TestCase):
    @staticmethod
    def get_hdf(name: str, number: int = 1) -> DashHDF:
        """Note: Relies on get_id and get_hdf"""
        hdf_id = HdfId(page='test', additional_classifier=name, number=number)
        hdf = DashHDF(hdf_id)
        return hdf

    def test_get_id(self):
        """Get the id of an HDF file"""
        hdf_id = HdfId(page='test', additional_classifier='testexp', number=1)
        self.assertEqual('test-testexp-1', hdf_id.uid)

    def test_get_hdf(self):
        """Get an hdf file"""
        hdf_id = HdfId(page='test', additional_classifier='testexp', number=1)
        hdf = DashHDF(hdf_id)
        self.assertEqual(hdf_id, hdf.hdf_id)

    def test_get_hdf_id(self):
        """Get a unique ID of HDF that can be used to load that HDF again"""
        hdf = self.get_hdf('id', number=1)
        id = hdf.id
        self.assertEqual({
            'uid': 'test-id-1'
        }, id)

    def test_init_from_id(self):
        hdf = self.get_hdf('init', number=1)
        id = hdf.id
        init_hdf = DashHDF(id)
        self.assertEqual(hdf.id, init_hdf.id)

    def test_read_data(self):
        """Read array from HDF file"""
        hdf = self.get_hdf('read')
        expected = np.array([1, 2, 3, 4])
        with HDFFileHandler(hdf.hdf_path, 'w') as f:
            f['test_read_data'] = expected
            g = f.require_group('subgroup')
            g['test_read_data'] = expected*2  # *2 just to differentiate written data

        read = hdf.get_data('test_read_data', subgroup=None)
        read_subgroup = hdf.get_data('test_read_data', subgroup='subgroup')

        self.assertTrue(np.all(expected == read))
        self.assertTrue(np.all(expected*2 == read_subgroup))

    def test_write_data(self):
        hdf = self.get_hdf('write')
        expected = np.array([1, 2, 3, 4])

        hdf.save_data(expected, name='test_write_data', subgroup=None)
        hdf.save_data(expected*2, name='test_write_subgroup_data', subgroup='subgroup')

        data = hdf.get_data('test_write_data', subgroup=None)
        subgroup_data = hdf.get_data('test_write_subgroup_data', subgroup='subgroup')
        self.assertTrue(np.all(expected == data))
        self.assertTrue(np.all(expected*2 == subgroup_data))

    def test_read_info(self):
        hdf = self.get_hdf('read')
        expected = dict(a='a', one=1)
        with HDFFileHandler(hdf.hdf_path, 'w') as f:
            f.attrs['test_read_info'] = json.dumps(expected)  # TODO: Might want to replace this with whatever function does the usual writing
            g = f.require_group('subgroup')
            g.attrs['test_read_info_subgroup'] = json.dumps(expected)

        info = hdf.get_info(name='test_read_info', subgroup=None)
        info_subgroup = hdf.get_info(name='test_read_info_subgroup', subgroup='subgroup')
        self.assertEqual(expected, info)
        self.assertEqual(expected, info_subgroup)

    def test_write_info(self):
        hdf = self.get_hdf('write')
        expected = dict(a='a', one=1)
        hdf.save_info(expected, name='test_write_info', subgroup=None)
        hdf.save_info(expected, name='test_write_info_subgroup', subgroup='subgroup')

        info = hdf.get_info('test_write_info', subgroup=None)
        info_subgroup = hdf.get_info('test_write_info_subgroup', subgroup='subgroup')
        self.assertEqual(expected, info)
        self.assertEqual(expected, info_subgroup)

