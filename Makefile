.PHONY: test repro_conrey89 theta_sweep

test:
	python -m pytest tests/ -v

repro_conrey89:
	mollifier repro conrey89

theta_sweep:
	mollifier theta-sweep conrey89
