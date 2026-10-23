from ev_energy import load_data, quality_report

PATH = "data/raw/ev_trip_energy.csv"


def test_load_shape_and_columns():
    df = load_data(PATH)
    assert df.shape[0] == 10151
    assert {"energy_kwh", "vehicle_id", "avg_speed_kmh"} <= set(df.columns)


def test_quality_report_runs():
    assert "outside_range" in quality_report(load_data(PATH)).columns


def test_clean_data_removes_bad_rows_and_columns():
    from ev_energy import clean_data
    df = load_data(PATH)
    out = clean_data(df)
    assert len(out) == len(df) - 1            # only the 435 V row is removed
    assert out["voltage_v"].max() < 400
    assert not {"speed_raw", "lat", "lon"} & set(out.columns)


def test_split_has_no_shared_vehicles():
    from ev_energy import clean_data, split_by_vehicle
    train, test = split_by_vehicle(clean_data(load_data(PATH)))
    assert set(train.vehicle_id).isdisjoint(set(test.vehicle_id))
