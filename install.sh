#pip install .
#black smudgy/
#ruff check smudgy/ --fix
#pytest -rs
rm -rf docs/_build
python3 -m sphinx -b html docs docs/_build/html -j 8