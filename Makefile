all: custom_ops

test: custom_ops
	py.test tests

.PHONY: docs
docs:
	$(MAKE) -C docs html

coverage:
	py.test --cov=. tests

visdom_server:
	python -m visdom.server -p 8080 &

clean:
	python setup.py clean
	rm -rf build ttools.egg-info dist .pytest_cache
