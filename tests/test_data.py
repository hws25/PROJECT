from ev_energy import load_data, quality_report

PATH = "data/raw/ev_trip_energy.csv"


def test_load_shape_and_columns():
    df = load_data(PATH)
    assert df.shape[0] == 10151
    assert {"energy_kwh", "vehicle_id", "avg_speed_kmh"} <= set(df.columns)


def test_quality_report_runs():
    assert "outside_range" in quality_report(load_data(PATH)).columns
