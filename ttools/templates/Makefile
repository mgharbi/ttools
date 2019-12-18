# Folders for I/O
DATA?=data
OUT?=output
CHKPTS?=checkpoints

VPORT=2001  # for Vidsom

# Launch a Visdom server
server:
	python -m visdom.server -port $(VPORT)

develop:
	python setup.py develop

install:
	python setup.py install

test:
	pytest tests


train/%:
	python scripts/train.py \
		--data $(DATA) \
		--val_data $(DATA) \
		--checkpoint_dir $(CHKPTS)/$* \
		--bs 1 \
		--num_worker_threads 4 \
		--lr 1e-4 \
		--port $(VPORT) \
		--config config/$*.yml \
		--env $*

eval/%:
	python scripts/eval.py \
		$(CHKPTS)/$* \
		$(DATA) \
		$(OUT)
