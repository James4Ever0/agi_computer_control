![Cybergod logo](propaganda/logos/cybergod_2.png)

https://github.com/Significant-Gravitas/Auto-GPT/assets/103997068/8e1cd6fe-c49d-4d2b-835d-0ffc9a5a458e

# Cybergod

[join discord group](https://discord.gg/eM5vezJvEQ)

[bilibili live streaming](http://live.bilibili.com/22228498)

Trained on [The Frozen Forest](https://huggingface.co/datasets/James4Ever0/the_frozen_forest), a [dataset](https://modelscope.cn/datasets/james4ever0/the_frozen_forest/summary) containing random keystrokes, mouse clicks and screen recordings.

the openai [universe](https://github.com/openai/universe) is using VNC, almost doing the same thing.

you can find some demo models from [there](https://github.com/openai/universe-starter-agent).

check out [SerpentAI](https://github.com/SerpentAI/SerpentAI)

but why bother? we can build these things in the same way.

human demonstrations are limited, but random keystrokes are infinite.

try to obtain infinite data as pretrained data, then fine-tune on human demonstrations.

---

键鼠真神是一种意识形态

cybergod is an ideology.

键鼠真神 又名cybergod 赛博真神

训练数据集为the frozen forest 随机敲键盘点鼠标 录屏

奖励函数 如果屏幕发生变化 奖励上一次行为

避免把系统关机 被锁在屏幕外面

避免机器卡死： 监测机器是否卡死 如果卡死那么自动外部重启 （重置状态，重新跑脚本）

连着WEBDAV一起刷新 有filelock

(直接取消lock权限)

---

looking for using docker for automation, or using some tty-like things for automation.

disable ubuntu system authentication?

---

make some server for vm to access to restart the webdav server when you have error.

---

agi_workspace happen to be in recycle bin. make sure we have the init files.

make sure we can restore our environments in every restart.

---

spice-protocol

found on utm.app, launch qemu and create spice unix socket.

https://github.com/Shells-com/spice

https://github.com/gnif/PureSpice

https://github.com/citrix-openstack-build/spice-html5

https://github.com/oetcxiaoliu/spice

https://github.com/TotallWAR/spice_protocol

remmina

---

掉盘问题： `cd .`

(建议直接换个盘 或者换C口的数据线 A口不稳定 或者把硬盘取出来更新固件？)

c口数据线观测中

---

to resolve the display resolution/mouse coordinate range matching issue, use pyautogui to get the resolution then capture display using that resolution (resize to it)

---

GPT4 is using MoE as its architecture.

---

the main objective of AGI is to create another version of itself.

---

ways of connection:

vnc, ssh, tty, tmux, hdmi capture & hid emulator, window capture and directed inputs (os specific)

---

the point is not making this exhaustive. it is about making some standard i/o and adapt to every situation.

---

改变开发思路：将功能和娱乐相结合

受众：游戏娱乐向 实用向

发布程序到steam平台

为此需要宣传、绘画设计等等

---

用elo进行打分 分高的可以在官网有较高的模型权重排名

---

technically this would not be a normal game. it is a metagame, which is the game of all games. it can play other games, play itself, even create itself.

---

devcontainer is useful for creating reproducible environments locally (if of the same architecture, like x86) or remotely (different architecture, like Apple M1).

---

because setting this up properly during development is a pain in the ass (in most time), let's pack it up into a docker container, for your safety.

if you want to release this and use it in production, you can refactor the code, configure platform specific dependencies and send it to devops.

---

devcontainer won't work as expected on windows 11 as we put our repo on external disk

---

your aim is too damn big! shall you begin to train some primitive neural network with functionality of only emitting and receiving ascii words, even just a single character like 'C'. get your hands dirty!

---

the basic docker service is just like havoc. it does not contain anything 'intelligent'. only 'life support'.

we plan to containerize chatdev/open-interpreter/autogpt. after that, we will combine the two, and create some 'capitalism' among multiple containers.

finally we will create some ever-evolving agent and use that as the building block for the megasystem.