rm agi_computer_control.7z
cd ..
7z a -snl agi_computer_control.7z agi_computer_control
7z d agi_computer_control.7z agi_computer_control/credentials.py
cd agi_computer_control
mv ../agi_computer_control.7z .
