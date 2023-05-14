systemctl enable redis-server
systemctl start redis-server
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs -iabc kill -s KILL abc
ps aux | grep bash | grep main_loop | grep -v grep | grep -v start | awk '{print $2}' | xargs -iabc kill -s KILL abc
nohup bash main_loop.sh &> /dev/null &