.PHONY: envtch  install shell

env:
	hatch env create

install:
	hatch run pip install -e .

shell:
	hatch shell
