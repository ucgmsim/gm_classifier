from setuptools import setup, find_packages

setup(
    name="gm_classifier",
    version="20.2.1",
    packages=find_packages(),
    url="",
    description="Ground motion record classifier",
    install_requires=["numpy", "pandas"],
    scripts=["gm_classifier/scripts/extract_features.py",
             "gm_classifier/scripts/gen_konno_matrices.py"
             "gm_classifier/scripts/run_predict.py"],
)