## Generate docs
```
cd doc
sphinx-apidoc -f -o . ../spsklearn/
sphinx-build -b html ./ ../_build
```