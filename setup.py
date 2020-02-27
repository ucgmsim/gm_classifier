from setuptools import setup, find_packages

setup(
    name="gm_classifier",
    version="20.2.1",
    packages=find_packages(),
    url="",
    description="Ground motion record classifier",
    install_requires=["numpy", "pandas", "matplotlib", "scipy", "tensorflow>=2.1.0"],
    scripts=["gm_classifier/scripts/extract_features.py",
             "gm_classifier/scripts/gen_konno_matrices.py",
             "gm_classifier/scripts/run_predict.py"],
    package_data={"gm_classifier": ["original_models/*", "tests/benchmark_tests/original_models/*.csv"]},
    include_package_data=True
)
