install:

	pip install --upgrade pip
	pip install -r requirements.txt

lint:

	pylint your_code_directory

all: install lint
