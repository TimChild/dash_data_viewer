"""
Any external data (i.e. from Dat, or file, or whatever) should be saved into a Dash HDF file and a store updated with
an ID of the Dash HDF, and maybe what was last updated in it?
This will save sending data back and forth to client, and I can interact with HDF files in a threadsafe/process safe way
with my HDFFileHandler (i.e. single process access, multi-thread if reading, single thread if writing)
"""
import h5py
import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict, field, InitVar
import os

from dacite import from_dict


from dat_analysis.hdf_file_handler import HDFFileHandler, GlobalLock

HDF_DIR = os.path.join(os.path.dirname(__file__), 'dash_hdfs')
os.makedirs(HDF_DIR, exist_ok=True)


@dataclass
class HdfId:
    page: InitVar[str] = None
    experiment: InitVar[str] = None
    number: InitVar[int] = None
    uid: Union[str, int] = None

    def __post_init__(self, page, experiment, number):
        if not self.uid:
            if number:
                uid = ''
                if page:
                    uid += f'{page}-'
                if experiment:
                    uid += f'{experiment}-'
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
        return from_dict(HdfId, d)

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
    def __init__(self, hdf_id: Union[dict, HdfId]):
        if not isinstance(hdf_id, HdfId):
            hdf_id = from_dict(HdfId, hdf_id)
        self.hdf_id: HdfId = hdf_id
        self.hdf_path = _get_hdf_path(hdf_id)
        self._using_context = False
        self._handler = None
        self._file: h5py.File = None

    def __enter__(self,  mode: str):
        """For context manager"""
        allowed_modes = ['r', 'r+']
        if mode not in allowed_modes:
            raise ValueError(f'{mode} not supported. Only {allowed_modes} allowed')
        if self._using_context:
            raise RuntimeError(f'Context manager already open for {self.hdf_id}')
        self._handler = HDFFileHandler(self.hdf_path, mode)
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
                if self._file.mode != 'r+':
                    raise RuntimeError(f'File is in {self._file.mode} mode, but requires {mode}')

    @classmethod
    def load_from_id(cls, hdf_id: dict):
        raise NotImplementedError

    def save_data(self, data: Union[np.ndarray, h5py.Dataset], name: str, subgroup: Optional[str] = None):
        "For saving arrays or dataset to HDF"
        self._check_file_open('r+')
        raise NotImplementedError

    def get_data(self, name: str, subgroup: Optional[str] = None) -> np.ndarray:
        "For getting array from HDF"
        self._check_file_open('r')
        raise NotImplementedError

    def get_dataset(self, name: str, subgroup: Optional[str] = None) -> h5py.Dataset:
        "For getting h5py.Dataset from HDF (note that the HDF must be kept open while using this)"
        self._check_file_open('r')
        raise NotImplementedError

    def save_info(self, info: Any, name: str, subgroup: Optional[str] = None):
        "For saving any other info/attributes to the HDF"
        self._check_file_open('r+')
        raise NotImplementedError







