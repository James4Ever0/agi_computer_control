# cd ..
# pytest --lf --lfnf=all --capture=tee-sys test_project.py

pytest --lf --lfnf=all --capture=tee-sys --log-level=DEBUG test_project.py

# no much difference.
# env BETTER_EXCEPTIONS=1 pytest --lf --lfnf=all --capture=tee-sys --log-level=DEBUG test_project.py

# pytest --lf --lfnf=all test_project.py
# pytest --lf --lfnf=all --capture=tee-sys test_project.py