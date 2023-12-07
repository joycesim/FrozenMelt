# this script runs all the VBRc post-processing

from dask.distributed import Client
from ridge_post_proc import process_all_runs
import shutil
import os
import sys

# Dask settings
threads_per_worker = 1
n_workers = 6

output_dir = 'output'
data_dir = 'data'
raw_data = 'data.tar.gz'

if __name__ == "__main__":

    if len(sys.argv) > 1:
        n_workers = int(sys.argv[1])

    if os.path.isdir(data_dir) is False:
        shutil.unpack_archive(raw_data, '.')

    c = Client(threads_per_worker=threads_per_worker, n_workers=n_workers)
    process_all_runs(data_dir, (500, 250), output_dir=output_dir)
    c.close()
