# from __future__ import annotations

import os

from dat_analysis.new_dat.dat_hdf import get_dat_from_exp_filepath, save_path_from_exp_path
from dat_analysis.hdf_file_handler import GlobalLock
import tempfile

import logging


logger = logging.getLogger(__name__)

global_lock = GlobalLock(os.path.join(tempfile.gettempdir(), 'dash_lock.lock'))


def get_dat(data_path):
    dat = None
    if data_path and os.path.exists(data_path):
        try:
            if os.path.exists(save_path_from_exp_path(data_path)):  # Already exists, so no need to lock while loading
                dat = get_dat_from_exp_filepath(data_path, overwrite=False)
            else:
                with global_lock:   # Only allow one thread to create the new path (the next will just load existing)
                    dat = get_dat_from_exp_filepath(experiment_data_path=data_path, overwrite=False)
        except Exception as e:
            logger.warning(f'Failed to load dat at {data_path}. \nRaised: \n{e}')
    else:
        logger.info(f'No file at {data_path}')
    return dat

