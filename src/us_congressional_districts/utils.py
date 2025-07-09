import pathlib


def get_state_fips_codes():
    return {
        "al": "01",
        "ak": "02",
        "az": "04",
        "ar": "05",
        "ca": "06",
        "co": "08",
        "ct": "09",
        "de": "10",
        "dc": "11",  # Washington DC
        "fl": "12",
        "ga": "13",
        "hi": "15",
        "id": "16",
        "il": "17",
        "in": "18",
        "ia": "19",
        "ks": "20",
        "ky": "21",
        "la": "22",
        "me": "23",
        "md": "24",
        "ma": "25",
        "mi": "26",
        "mn": "27",
        "ms": "28",
        "mo": "29",
        "mt": "30",
        "ne": "31",
        "nv": "32",
        "nh": "33",
        "nj": "34",
        "nm": "35",
        "ny": "36",
        "nc": "37",
        "nd": "38",
        "oh": "39",
        "ok": "40",
        "or": "41",
        "pa": "42",
        "ri": "44",
        "sc": "45",
        "sd": "46",
        "tn": "47",
        "tx": "48",
        "ut": "49",
        "vt": "50",
        "va": "51",
        "wa": "53",
        "wv": "54",
        "wi": "55",
        "wy": "56",
    }


def state_abbr_from_fips():
    return {v: k for k, v in get_state_fips_codes().items()}


def get_data_directory() -> pathlib.Path:
    """
    Determines the absolute path to the 'us-congressional-districts/inputs' directory,
    assuming the script is located within 'us-congressional-districts/src/us_congressional_districts'.
    """
    script_path = pathlib.Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent
    inputs_dir = repo_root / "data"

    return inputs_dir


def main():
    inputs_directory = get_data_directory()

    print(f"The inputs directory is: {inputs_directory}")
    print(f"Does the inputs directory exist? {inputs_directory.exists()}")


if __name__ == "__main__":
    main()
