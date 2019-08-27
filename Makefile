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

distribution:
	python setup.py sdist bdist_wheel
	twine check dist/*

test_upload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload_distribution:
	twine upload dist/*
