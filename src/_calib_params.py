# calibration dates used in the paper
CALIB_DATES = ("20241219", "20250416", "20250620", "20250702")

def get_calibration_params(date):
    if date == "20230215":
        params = {"al": 0.568, "lam": 9.68, "nu": 0.572, "c": 8.1e-3}
    elif date == "20240716":
        params = {
            "al": 0.5 + 9.7163e-2,
            "nu": 8.2733e-1,
            "c": 3.6647e-3,
            "lam": 1.9678e1,
        }
    elif date == "20241219":
        params = {
            "al": 0.5 + 9.6384e-2,
            "nu": 7.6196e-1,
            "c": 4.5572e-3,
            "lam": 6.25586,
        }
    elif date == "20250416":
        params = {
            "al": 0.5 + 0.2908,
            "lam": 5,
            "nu": 1.2942,
            "c": 0.018977,
        }
    elif date == "20250620":
        params = {
            "al": 0.5 + 2.3267e-2,
            "lam": 6.3976,
            "nu": 3.6756e-1,
            "c": 2.3084e-3,
        }
    elif date == "20250702":
        params = {
            "al": 0.5 + 2.4157e-2,
            "lam": 5.6395,
            "nu": 3.6689e-1,
            "c": 2.8158e-3,
        }
    else:
        raise ValueError(f"check value for {date}.")
    return params
