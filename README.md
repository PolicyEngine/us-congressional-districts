# Congressional Districts Estimation

After changing directories to the repo root and installing the package with

```
uv pip install -e .
uv pip install policyengine-us # There seems to be a dependency issue requiring -us to be installed after.
```

You can start the congressional district reweighting procedure with the command:
```
us-congressional-districts
```
The inputs are in subfolders of './data/inputs'. The code in ./src can generate them
by downloading from the appropriate sources, but there isn't a `make data` step
defined yet that does that

The outputs are in './data/output'
