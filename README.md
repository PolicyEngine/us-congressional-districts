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

The outputs are in './data/output'
