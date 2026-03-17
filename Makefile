.PHONY: build publish test deploy


build:
	rm -rf dist
	uvx --from build pyproject-build --installer uv

test:
	uv run pytest tests/

publish: build
	uvx twine upload dist/*

deploy:
	npx wrangler pages deploy
