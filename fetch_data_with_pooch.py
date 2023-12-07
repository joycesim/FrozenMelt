try:
    import pooch
except ImportError as e:
    print("This functionality requires pooch. Please run `pip install pooch` and try again.")
    raise e
import shutil
import os

data = {
    'post_processing': ['https://zenodo.org/records/10288494/files/PostProcessed.zip',
                        "md5:2c960421efee652ec122b2632aae91cf",
                        os.path.join(".", "post_processing")],
    'seismic_post': ['https://zenodo.org/records/10288494/files/VTU.zip?download=1',
                     'md5:5bc28ec885e2a212a163112fb08f2617',
                     os.path.join(".","seismic_post_processing", "data")],
}

top_data_dir = "FrozenMeltData"

if __name__ == '__main__':

    if os.path.isdir(top_data_dir) is False:
        os.mkdir(top_data_dir)

    for fname, info in data.items():
        unpacked_dirname = os.path.join(top_data_dir, info[0].split("/")[-1].split(".")[0])

        if os.path.isdir(unpacked_dirname) is False:
            zenodo_url = info[0]
            hash = info[1]
            full_archive = pooch.retrieve(zenodo_url, hash, path=top_data_dir)
            shutil.unpack_archive(full_archive, extract_dir=top_data_dir)

        for fi in os.listdir(unpacked_dirname):
            src = os.path.join(unpacked_dirname, fi)
            dest = os.path.join(info[2], fi)
            shutil.move(src, dest)

        os.rmdir(unpacked_dirname)
