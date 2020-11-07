import pickle
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

import gm_classifier as gmc


@pytest.mark.parametrize("record_name", ["20190131_155600_SEAS_20", "20120901_161043_RQGS", "20190704_083147_OXZ_10"])
def test_v1a_feature_calc(record_name: str):
    cur_dir = Path(__file__).parent

    # Run the feature extraction
    record_ffp = cur_dir / "bench_data" / f"{record_name}.V1A"
    input_data, add_data = gmc.records.process_record(str(record_ffp), str(cur_dir / "konno_matrices"))

    # Load the benchmark data
    with open(cur_dir / "bench_data" / f"{record_name}.pickle", "rb") as f:
        bench_data = pickle.load(f)
    bench_input_data = bench_data["input_data"]
    bench_add_data = bench_data["add_data"]

    for cur_key in ["1", "2", "v"]:
        cur_input_data = pd.Series(input_data[cur_key]).sort_index()
        cur_bench_input_data = pd.Series(bench_input_data[cur_key]).sort_index()

        assert np.allclose(cur_input_data.values, cur_bench_input_data.values)

    assert input_data["record_id"] == bench_input_data["record_id"]
    assert input_data["event_id"] == bench_input_data["event_id"]
    assert input_data["station"] == bench_input_data["station"]

    assert add_data["p_wave_ix"] == bench_add_data["p_wave_ix"]
    assert add_data["s_wave_ix"] == bench_add_data["s_wave_ix"]

    return


@pytest.mark.parametrize("record_name", ["20190704_083147_OXZ_10"])
def test_mseed_feature_calc(record_name: str):
    cur_dir = Path(__file__).parent

    # Run the feature extraction
    record_ffp = cur_dir / "bench_data" / f"{record_name}.mseed"
    input_data, add_data = gmc.records.process_record(str(record_ffp), str(cur_dir / "konno_matrices"))

    # Load the benchmark data
    with open(cur_dir / "bench_data" / f"{record_name}.pickle", "rb") as f:
        bench_data = pickle.load(f)
    bench_input_data = bench_data["input_data"]
    bench_add_data = bench_data["add_data"]

    for cur_key in ["1", "2", "v"]:
        cur_input_data = pd.Series(input_data[cur_key]).sort_index()
        cur_bench_input_data = pd.Series(bench_input_data[cur_key]).sort_index()

        assert np.allclose(cur_input_data.values, cur_bench_input_data.values, rtol=1e-1)

    assert input_data["record_id"] == bench_input_data["record_id"]
    assert input_data["event_id"] == bench_input_data["event_id"]
    assert input_data["station"] == bench_input_data["station"]

    assert add_data["p_wave_ix"] == bench_add_data["p_wave_ix"]
    assert add_data["s_wave_ix"] == bench_add_data["s_wave_ix"]

    return


if __name__ == '__main__':
    test_v1a_feature_calc("20190131_155600_SEAS_20")


