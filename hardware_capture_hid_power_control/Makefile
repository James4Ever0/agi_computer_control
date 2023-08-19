main: test_control.py
	${PYTHON} test_control.py

test_control.py: test_control.py.j2 $(addprefix ../, ${RENDERED_CODE})
	${PYTHON} ../render_python_code.py test_control.py