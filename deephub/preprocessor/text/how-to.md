## End to end pipeline for one dataset
Example command:
```
python -m deep_preprocessor.text 
path/to/train/file_pattern 
path/to/other/file_pattern 
--output-dir some/out/dir
--min-count-term 1 
--max-row-length 100
```

Type `python -m deep_preprocessor.text 
--help` for details.
