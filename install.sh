pip install .
black . 
ruff check . --fix 
pytest -rs
rm -rf docs/_build
python3 -m sphinx -b html docs docs/_build/html