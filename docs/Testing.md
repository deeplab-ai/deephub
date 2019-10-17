## Testing Procedure

### Execute tests

__deephub__ is shipped with multiple tests ranging from unit-test to full end-to-end tests. Tests are primarly
written in `pytest` but there are also legacy tests written in `unittest`. To distinguish the nature of tests and
permit an effortless continue integration, tests are marked as:

* **slow**: Any test that takes more than `~2 seconds` is considered a slow test.
* **gpu_only**: Tests that can only be run on GPU are marked
* **toy_data**: These tests depend on toy data to run.

To run the tests you can use the `setup.py` subcommand `pytest` and use `--addopts` for extra pytest options.

For example to run tests that are not slow and do not need gpu:
```
$ python setup.py pytest --addopts ' -m "not slow and not gpu_only"'
```

If you want to run all the tests that are included in the CI pipeline you can use the special subcommand `ci_tests`.
This command will execute only fast unit tests that do not depend on GPU or downloadable resources and will 
write a JUnit type report under `test-reports` folder:
```
$ python setup.py ci_tests
```