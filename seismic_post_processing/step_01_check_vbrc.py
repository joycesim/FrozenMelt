import os
import shutil
import pooch
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        download_vbrc = bool(int(sys.argv[1]))
    else:
        download_vbrc = False

    vbr_path = os.environ.get('vbrdir', None)
    loc_vbr_exists = os.path.isdir(os.path.abspath('vbr-1.1.2'))
    if loc_vbr_exists:
        download_vbrc = False
        print("Local copy of vbr-1.1.2 found in this directory and will be used. Delete it "
              "to force a new download or to use a local copy specified by the vbrdir "
              "environment variable.")
    elif vbr_path is None and download_vbrc is False:
        raise FileNotFoundError("Could not find local copy of the VBRc, either set the vbrdir "
                                "environment variable or call this script with "
                                "`python step_01_check_vbrc.py 1` to download a copy.")
    elif vbr_path is not None:
        if download_vbrc is True:
            print(f"Found VBRc installation in {vbr_path}, but will download a new copy. "
                  f"Use `python step_01_check_vbrc.py 0` to use the local copy")
        else:
            print(f"Found VBRc installation in {vbr_path}. Use `python step_01_check_vbrc.py 1` "
                  f"to download v1.1.2 to use here.")

    if download_vbrc:
        vbr_1pt1pt2 = "https://github.com/vbr-calc/vbr/archive/refs/tags/v1.1.2.tar.gz"
        fname = pooch.retrieve(vbr_1pt1pt2, None)
        shutil.unpack_archive(fname)
        if os.path.isfile(os.path.join("vbr-1.1.2", "vbr_init.m")):
            os.remove(fname)
        else:
            msg = "Download failed in an unexpected way. See README.md for manual installation of VBRc."
            raise FileNotFoundError(msg)
