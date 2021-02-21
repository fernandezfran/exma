run `tox` in the main folder to test all cases.

note that if you run `python3 -m unittest discover` from this folder, the statistic test and writer_and_reader will fail because they both write and read files that must be runned from the main folder.
