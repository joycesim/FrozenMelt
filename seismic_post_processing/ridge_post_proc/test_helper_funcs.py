# these tests are just for some of the helper functions in ridge_post_proc.main
# pytest is required to run (which you can do so with `$pytest ridge_post_proc`)
from .main import _load_run_info_data, _RunInfoData
import numpy as np


def test_save_load_RunInfoData(tmp_path):
    # check that the intermediate hdf5 metadata file works
    nx = 5
    nz = 10
    vbrc_files = {
        "baseline": f"baseline_VBRc_output.mat",
        "separate": f"separate_VBRc_output.mat",
        "separate_secondary": f"separate_VBRc_output_secondary.mat",
    }
    fname = str(tmp_path / 'test_rid.hdf5')
    rid = _RunInfoData(x=np.ones((nx,)),
                       z=np.ones((nz,)),
                       phi=np.ones((nz, nx)),
                       degF=np.ones((nz, nx)),
                       refert=np.ones((nz, nx)),
                       fname=fname,
                       length_unit='km',
                       buff_size=(8, 16),
                       U=2,
                       K=7,
                       vbrc_output_files=vbrc_files, )

    rid.save(fname)

    rid_in = _load_run_info_data(fname)
    for ky in ("x", "z", "phi", "degF", "refert"):
        assert np.all(getattr(rid_in, ky) == getattr(rid, ky))

    for ky in ("fname", "length_unit", "buff_size", "U", "K", "vbrc_output_files"):
        assert getattr(rid_in, ky) == getattr(rid, ky)
