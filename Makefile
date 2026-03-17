.PHONY: build publish test


build:
	rm -rf dist
	uvx --from build pyproject-build --installer uv

test:
	uv run pytest tests/

publish: build
	uvx twine upload dist/*
