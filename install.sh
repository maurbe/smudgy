# convenience script for full installation, linting, testing and docs building
pip install -e .
black smudgy/
ruff check smudgy/ --fix
pytest -rs
rm -rf docs/_build
python3 -m sphinx -b html docs docs/_build/html -j 8