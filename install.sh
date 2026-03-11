pip install .
black . 
ruff check . -- fix 
pytest -rs
python3 -m sphinx -b html docs docs/_build/html