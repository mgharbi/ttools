all: custom_ops test coverage

test:
	py.test tests

coverage:
	py.test --cov=. tests

custom_ops:
	$(MAKE) -C $@ submodules
	$(MAKE) -C $@

.PHONY: custom_ops

