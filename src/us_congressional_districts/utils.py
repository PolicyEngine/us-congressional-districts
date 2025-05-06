import pathlib


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
