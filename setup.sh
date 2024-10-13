#!/bin/bash
conda env create -f environment.yml
cd llm2vec
pip install --editable .
cd ../mteb
pip install --editable .
cd ../

