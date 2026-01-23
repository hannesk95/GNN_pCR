import dicom2nifti
import os
from tqdm import tqdm
from glob import glob
import dicom2nifti
import os
import subprocess
import shutil
from pathlib import Path
from typing import List

def dicom_to_nifti_dcm2niix(
    dicom_dir: str | Path,
    output_nifti_path: str | Path,
    gzip: bool = True,
    overwrite: bool = False,
    extra_args: List[str] | None = None,
) -> List[Path]:
    """
    Convert a DICOM directory to NIfTI using dcm2niix.

    Parameters
    ----------
    dicom_dir : str or Path
        Directory containing DICOM files.
    output_nifti_path : str or Path
        Desired output NIfTI file path (used to derive output dir + filename).
    gzip : bool, default=True
        Whether to gzip the output (.nii.gz).
    overwrite : bool, default=False
        Whether to overwrite existing output directory.
    extra_args : list of str, optional
        Additional dcm2niix command-line arguments.

    Returns
    -------
    List[Path]
        List of generated NIfTI files.
    """

    dicom_dir = Path(dicom_dir).expanduser().resolve()
    output_nifti_path = Path(output_nifti_path).expanduser().resolve()

    if not dicom_dir.is_dir():
        raise ValueError(f"DICOM directory does not exist: {dicom_dir}")

    out_dir = output_nifti_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # dcm2niix uses filename *pattern*, not full path
    filename_stem = output_nifti_path.stem
    if filename_stem.endswith(".nii"):
        filename_stem = filename_stem[:-4]

    if overwrite:
        for f in out_dir.glob(f"{filename_stem}*"):
            f.unlink()

    cmd = [
        "dcm2niix",
        "-z", "y" if gzip else "n",
        "-f", filename_stem,
        "-o", str(out_dir),
        str(dicom_dir),
    ]

    if extra_args:
        cmd[1:1] = extra_args

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "dcm2niix failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    # Collect generated files
    # suffix = ".nii.gz" if gzip else ".nii"
    # nifti_files = sorted(out_dir.glob(f"{filename_stem}*{suffix}"))

    # if not nifti_files:
    #     raise RuntimeError(
    #         f"dcm2niix finished successfully but no NIfTI files were found.\n"
    #         f"STDOUT:\n{result.stdout}"
    #     )

    # return nifti_files

failed_conversions = []

for patient_dir in tqdm(glob("/media/johannes/HDD_8.0TB/ISPY2_new/ISPY2-*")):
    print(f"Processing patient directory: {patient_dir}")

    for study_dir in os.listdir(patient_dir):
        study_dir = os.path.join(patient_dir, study_dir)      

        if not os.path.isdir(study_dir):
            continue        

        if "DCE" in study_dir:
            output_file = study_dir + ".nii.gz"
            # if os.path.exists(output_file):
            #     pass
            #     # print(f"Output file {output_file} already exists. Skipping conversion.")
            # else:            
            try:
                # dicom2nifti.dicom_series_to_nifti(study_dir, output_file)
                nifti_files = dicom_to_nifti_dcm2niix(study_dir, output_file)                

            except Exception as e:
                print(patient_dir)
                failed_conversions.append(patient_dir.split("/")[-1])
                print(f"Error converting {study_dir}: {e}\n\n")
        
for file in glob("/media/johannes/HDD_8.0TB/ISPY2_new/ISPY2-*/*.json"):
    os.remove(file)

    