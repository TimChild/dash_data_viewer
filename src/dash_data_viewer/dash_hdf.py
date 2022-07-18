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


from dat_analysis.hdf_file_handler import HDFFileHandler, GlobalLock
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
            if number:
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


class DashHDF:
    def __init__(self, hdf_id: Union[dict, HdfId], mode='r'):
        if not isinstance(hdf_id, HdfId):
            hdf_id.setdefault('page', None)
            hdf_id.setdefault('additional_classifier', None)
            hdf_id.setdefault('number', None)
            hdf_id = HdfId(**hdf_id)
        self.mode = mode  # Used when opening file in self.__enter__
        self.hdf_id: HdfId = hdf_id
        self.hdf_path = _get_hdf_path(hdf_id)
        self._using_context = False
        self._handler = None
        self._file: h5py.File

    @property
    def id(self):
        return self.hdf_id.asdict()

    @wraps(h5py.File.get)
    def get(self, name, default=None, getclass=None, getlink=None):
        self._check_file_open('r')
        return self._file.get(name, default=default, getclass=getclass, getlink=getlink)

    def __enter__(self) -> Any:
        """For context manager"""
        allowed_modes = ['r', 'r+', 'w']
        if self.mode not in allowed_modes:
            raise ValueError(f'{self.mode} not supported. Only {allowed_modes} allowed')
        if self._using_context:
            raise RuntimeError(f'Context manager already open for {self.hdf_id}')
        self._handler = HDFFileHandler(self.hdf_path, self.mode)
        self._using_context = True
        self._file = self._handler.new()
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For context manager"""
        try:
            self._handler.previous()
        finally:
            self._using_context = False

    def _check_file_open(self, mode=None):
        if not self._using_context:
            raise RuntimeError(f'Context manager not open. Remember to use context manager to open file before accessing (e.g. with DashHDF("r+") as f:\n\t...')
        if mode is not None:
            if mode == 'r':
                pass  # Doesn't matter what mode hdf is in
            elif mode == 'r+':
                if self._file.mode not in ['r+', 'w']:
                    raise RuntimeError(f'File is in {self._file.mode} mode, but requires {mode}')
            elif mode == 'w':
                if self._file.mode != 'w':
                    raise RuntimeError(f'File is in {self._file.mode} mode, but requires {mode}')

    def require_group(self, name: str):
        self._check_file_open(mode='r+')
        return self._file.require_group(name)

    def save_data(self, data: Union[np.ndarray, h5py.Dataset], name: str, subgroup: Optional[str] = None):
        """For saving arrays or dataset to HDF"""
        if isinstance(data, np.ndarray):
            self._check_file_open('r+')
            subgroup = subgroup if subgroup else '/'
            group = self._file.require_group(subgroup)
            set_data(group, name, data)

    def _get_dataset(self, name, subgroup) -> h5py.Dataset:
        self._check_file_open('r')
        subgroup = subgroup if subgroup else '/'
        group = self._file[subgroup]
        dataset = get_dataset(group, name)
        return dataset

    def get_data(self, name: str, subgroup: Optional[str] = None) -> np.ndarray:
        """For getting array from HDF"""
        dataset = self._get_dataset(name, subgroup)
        return dataset[:]

    def get_dataset(self, name: str, subgroup: Optional[str] = None) -> h5py.Dataset:
        """For getting h5py.Dataset from HDF (note that the HDF must be kept open while using this)"""
        return self._get_dataset(name, subgroup)

    def save_info(self, info: Any, name: str, subgroup: Optional[str] = None):
        """For saving any other info/attributes to the HDF"""
        self._check_file_open('r+')
        subgroup = subgroup if subgroup else '/'
        group = self._file.require_group(subgroup)
        set_attr(group, name, info, dataclass=None)

    def get_info(self, name: str, subgroup: Optional[str] = None) -> Any:
        """For getting info/attribute from HDF"""
        self._check_file_open('r')
        subgroup = subgroup if subgroup else '/'
        group = self._file.require_group(subgroup)
        info = get_attr(group, name, default=None, check_exists=True, dataclass=None)
        return info







