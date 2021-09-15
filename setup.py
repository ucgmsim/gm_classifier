from setuptools import setup, find_packages

setup(
    name="gm_classifier",
    version="20.2.1",
    packages=find_packages(),
    url="",
    description="Ground motion record classifier",
    install_requires=["tensorflow", "pandas", "scipy", "obspy", "numpy",
                      "matplotlib", "seaborn", "h5py", "scikit-learn", "tqdm"],
#    scripts=["gm_classifier/scripts/extract_features.py",
#             "gm_classifier/scripts/gen_konno_matrices.py",
#             "gm_classifier/scripts/predict.py"],
#    package_data={"gm_classifier": ["model/*"]},
    include_package_data=True
)
