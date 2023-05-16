the openai [universe](https://github.com/openai/universe) is using VNC, almost doing the same thing.

you can find some demo models from [there](https://github.com/openai/universe-starter-agent).

check out [SerpentAI](https://github.com/SerpentAI/SerpentAI)

but why bother? we can build these things in the same way.

human demonstrations are limited, but random keystrokes are infinite.

try to obtain infinite data as pretrained data, then fine-tune on human demonstrations.

----

键鼠真神 又名cybergod 赛博真神

训练数据集为the frozen forest 随机敲键盘点鼠标 录屏

奖励函数 如果屏幕发生变化 奖励上一次行为

避免把系统关机 被锁在屏幕外面

避免机器卡死： 监测机器是否卡死 如果卡死那么自动外部重启 （重置状态，重新跑脚本）


连着WEBDAV一起刷新 有filelock

(直接取消lock权限)
----

looking for using docker for automation, or using some tty-like things for automation.

disable ubuntu system authentication?

----

make some server for vm to access to restart the webdav server when you have error.

----

agi_workspace happen to be in recycle bin. make sure we have the init files.

make sure we can restore our environments in every restart.

----

spice-protocol

found on utm.app, launch qemu and create spice unix socket.

https://github.com/Shells-com/spice

https://github.com/gnif/PureSpice

https://github.com/citrix-openstack-build/spice-html5

https://github.com/oetcxiaoliu/spice

https://github.com/TotalWAR/spice_protocol

remmina

