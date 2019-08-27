all: custom_ops

test: custom_ops
	py.test tests

.PHONY: docs
docs:
	$(MAKE) -C docs html

coverage:
	py.test --cov=. tests

clean:
	python setup.py clean
	rm -rf build ttools.egg-info dist .pytest_cache
