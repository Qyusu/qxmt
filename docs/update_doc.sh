rm -rf _build
poetry run sphinx-apidoc -f -o ../docs/source ../qxmt
poetry run make html
