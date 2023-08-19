.PHONY: test

# may borrow system check code from ies.

PLATFORM := $(shell python -c "import os; print(os.name)")
ifeq (${PLATFORM}, )
PLATFORM := $(shell python3 -c "import os; print(os.name)") # executed on macos
endif

ifeq (${PLATFORM}, nt)
OS_TYPE = windows
else
OS_TYPE = macos
endif

PYTHON_ENV = -X utf8=1

ifeq (${OS_TYPE}, macos)
CONDA_ENV = rosetta
PYTHON = /usr/bin/python3
PIP = ${PYTHON} -m pip
else
CONDA_ENV = cplex
PYTHON = python ${PYTHON_ENV}
PIP = gsudo ${PYTHON} -m pip
endif

#### ALWAYS REMEMBER TO EXPORT USEFUL VARIABLES ####
export OS_TYPE PLATFORM PYTHON PYTHON_ENV CONDA_ENV
#### ALWAYS REMEMBER TO EXPORT USEFUL VARIABLES ####


RENDERED_CODE = conscious_struct.py hid_utils.py
RENDER_UTILS = jinja_utils.py pyright_utils.py
UTILS = log_utils.py ${RENDER_UTILS}
UTILS_SYNC_DIR = ../jubilant-adventure2/microgrid_base/
# shall you dump log to file, not to display it here

export RENDERED_CODE

test: ${UTILS} ${RENDERED_CODE} test/test_project.py
	cd test && ${PYTHON} -m pytest --lf --lfnf=all --capture=tee-sys test_project.py
	# cd test && ${PYTHON} -m pytest --lf --lfnf=all --capture=tee-sys --log-level=DEBUG test_project.py

${RENDERED_CODE}: $(addsuffix .j2, ${RENDERED_CODE}) ${RENDER_UTILS}
	${PYTHON} render_python_code.py $@

${UTILS}: $(addprefix  ${UTILS_SYNC_DIR}, ${UTILS})
	bash sync_utils.sh $@ ${UTILS_SYNC_DIR}

setup:
	${PIP} install -r requirements.txt

KL2XKS.json: hid_utils.py
	${PYTHON} hid_utils.py

software_interface: ${RENDERED_CODE}
	${MAKE} -e -C software_capture_hid_control

hardware_interface: ${RENDERED_CODE}
	${MAKE} -e -C hardware_capture_hid_power_control