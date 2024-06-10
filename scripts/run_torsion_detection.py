import subprocess

from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


"""
Run torsion detection on all files in a directory. If segmentation fault occurs, repeat the file.
"""


if __name__ == '__main__':
    files = [p for p in Path('/path/to/torsion_format/knee/severe').iterdir()]
    n_files = len(files)
    aug = 'mr'

    i = 0
    while i < n_files:
        try:
            subprocess.run(['python', 'torsion_detection_nnunet.py', '-file', str(files[i]), '-aug', aug], check=True)
        except subprocess.CalledProcessError as e:
            if e.returncode == -11:
                print(f'Segmentation fault at iteration {i}, file {files[i]}. Repeating.')
            else:
                print(e.returncode)
                raise e
        else:
            i += 1
            print(f'Processed {i}/{n_files} files')
