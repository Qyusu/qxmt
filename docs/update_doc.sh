rm -rf _build
poetry run sphinx-apidoc -o ../docs/source ../qxmt
poetry run make html
