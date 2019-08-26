all: custom_ops

test: custom_ops
	py.test tests

coverage:
	py.test --cov=. tests

custom_ops:
	$(MAKE) -C $@ submodules
	$(MAKE) -C $@

clean:
	python setup.py clean
	rm -rf build ttools.egg-info

.PHONY: custom_ops
