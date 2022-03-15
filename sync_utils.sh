# cp ../jubilant-adventure2/microgrid_base/log_utils.py .
# cp ../jubilant-adventure2/microgrid_base/jinja_utils.py .
# sed -i "s/microgrid/agi_computer_control/g" log_utils.py

cp $2$1 .
# echo $1
if [[ "$1" == "log_utils.py" ]]; then
    sed -i "s/microgrid/agi_computer_control/g" $1
fi