all: atis.json wsj2.json

atis.json: atis.pcfg
	PYTHONPATH=../..:$$PYTHONPATH python3 make_pcfg.py atis.pcfg TOP atis.json

wsj2.json: wsj2.pcfg
	PYTHONPATH=../..:$$PYTHONPATH python3 make_pcfg.py wsj2.pcfg ROOT_0 wsj2.json
