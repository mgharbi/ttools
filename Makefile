all: custom_ops

test: custom_ops
	py.test tests

coverage:
	py.test --cov=. tests

custom_ops:
	$(MAKE) -C $@ submodules
	$(MAKE) -C $@

clean:
	$(MAKE) -C custom_ops clean 

.PHONY: custom_ops
