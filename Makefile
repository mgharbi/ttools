test:
	py.test tests

coverage:
	py.test --cov=. tests

custom_ops:
	$(MAKE) -C $@

.PHONY: custom_ops

