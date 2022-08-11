"""
Any external data (i.e. from Dat, or file, or whatever) should be saved into a Dash HDF file and a store updated with
an ID of the Dash HDF, and maybe what was last updated in it?
This will save sending data back and forth to client, and I can interact with HDF files in a threadsafe/process safe way
with my HDFFileHandler (i.e. single process access, multi-thread if reading, single thread if writing)
"""
# from __future__ import annotations
import h5py
import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict, field, InitVar
import os
from functools import wraps
from contextlib import AbstractContextManager

from dacite import from_dict


from dat_analysis.hdf_file_handler import HDFFileHandler, GlobalLock, HDF
from dat_analysis.hdf_util import NotFoundInHdfError, set_attr, get_attr, set_data, get_data, get_dataset

HDF_DIR = os.path.join(os.path.dirname(__file__), 'dash_hdfs')
os.makedirs(HDF_DIR, exist_ok=True)


@dataclass
class HdfId:
    page: InitVar[str] = None  # To keep track of which page is creating which HDFs (prevents overlap between pages)
    additional_classifier: InitVar[str] = None  # e.g. prevent overlap when comparing between experiments
    number: InitVar[int] = None  # Usually datnum
    uid: Union[str, int] = None  # Optionally specify the whole unique ID in which case other InitVars are ignored

    def __post_init__(self, page, additional_classifier, number):
        if not self.uid:
            if number is not None:
                uid = ''
                if page:
                    uid += f'{page}-'
                if additional_classifier:
                    uid += f'{additional_classifier}-'
                uid += f'{number}'
            else:
                uid = self._get_next_id()
            self.uid = uid
        else:
            self.uid = self._sanitize(self.uid)

    def asdict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        """Get the HdfId object back from its dict representation"""
        return from_dict(HdfId, d)

    @classmethod
    def from_id(cls, id: dict):
        """Alias for 'from_dict' """
        return cls.from_dict(id)

    @property
    def filename(self) -> str:
        return f'{self.uid}.h5'

    @staticmethod
    def _sanitize(uid) -> str:
        """Make sure uid is filename safe"""
        keep_chars = ('-', '_')
        return "".join([c for c in uid if c.isalnum() or c in keep_chars]).rstrip()

    @staticmethod
    def _get_next_id() -> str:
        """Check in HDF folder for the next free temp_number to assign
        e.g. temp_1.h5

        Should be prefixed with temp so that it is easy to know which don't benefit from living longer than one session
        (i.e. specified ones might actually be reused and benefit from being saved longer)
        """
        raise NotImplementedError


def _get_hdf_path(hdf_id: Union[dict, HdfId]) -> str:
    """
    Make sure a HDF exists for id given, then return path to it
    Args:
        hdf_id ():

    Returns:

    """
    if not isinstance(hdf_id, HdfId):
        hdf_id = from_dict(HdfId, hdf_id)
    path = os.path.join(HDF_DIR, hdf_id.filename)
    if not os.path.isfile(path):
        with HDFFileHandler(path, 'w') as f:
            pass
    return path


class DashHDF(HDF):
    def __init__(self, hdf_id: Union[dict, HdfId]):
        if not isinstance(hdf_id, HdfId):
            hdf_id.setdefault('page', None)
            hdf_id.setdefault('additional_classifier', None)
            hdf_id.setdefault('number', None)
            hdf_id = HdfId(**hdf_id)
        self.hdf_id: HdfId = hdf_id
        self.hdf_path = _get_hdf_path(hdf_id)
        super().__init__(self.hdf_path)

    @property
    def id(self):
        return self.hdf_id.asdict()

    def save_data(self, data: Union[np.ndarray, h5py.Dataset], name: str, subgroup: Optional[str] = None):
        """For saving arrays or dataset to HDF"""
        if isinstance(data, np.ndarray):
            with self.hdf_write as f:
                subgroup = subgroup if subgroup else '/'
                group = f.require_group(subgroup)
                set_data(group, name, data)

    def _get_dataset(self, name, subgroup) -> h5py.Dataset:
        with self.hdf_read as f:
            subgroup = subgroup if subgroup else '/'
            group = f[subgroup]
            dataset = get_dataset(group, name)
        return dataset

    def get_data(self, name: str, subgroup: Optional[str] = None) -> np.ndarray:
        """For getting array from HDF"""
        with self.hdf_read as f:
            subgroup = subgroup if subgroup else '/'
            group = f[subgroup]
            dataset = get_dataset(group, name)[:]
        return dataset

    def save_info(self, info: Any, name: str, subgroup: Optional[str] = None):
        """For saving any other info/attributes to the HDF"""
        with self.hdf_write as f:
            subgroup = subgroup if subgroup else '/'
            group = f.require_group(subgroup)
            set_attr(group, name, info, dataclass=None)

    def get_info(self, name: str, subgroup: Optional[str] = None) -> Any:
        """For getting info/attribute from HDF"""
        with self.hdf_read as f:
            subgroup = subgroup if subgroup else '/'
            group = f.require_group(subgroup)
            info = get_attr(group, name, default=None, check_exists=True, dataclass=None)
        return info







