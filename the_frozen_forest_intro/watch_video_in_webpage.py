# warning! videos from bilibili must be viewed with custom userscripts
# you can tweak that, or use other videos to replace it.

# whatever. just trying.
from DrissionPage.easy_set import set_paths

BROWSER_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
# DOWNLOAD_PATH = r""
USER_DATA_PATH = r"F:\MicrosoftEdgeUserData"
# CACHE_PATH = r""
# you may have to install extensions yourself.

set_paths(
    browser_path=BROWSER_PATH,
    # download_path=DOWNLOAD_PATH,
    user_data_path=USER_DATA_PATH,
    # cache_path=CACHE_PATH,
)

# no sound!
# 关闭静音开播

from DrissionPage import ChromiumPage #, ChromiumOptions

# co = ChromiumOptions()
# co.set_mute(True)
# co.set_mute(False)

# userscripts to fix bilibili ads, for babysitting
# https://greasyfork.org/zh-CN/scripts/467511-bilibili-%E5%9C%A8%E6%9C%AA%E7%99%BB%E5%BD%95%E7%9A%84%E6%83%85%E5%86%B5%E4%B8%8B%E8%87%AA%E5%8A%A8%E5%B9%B6%E6%97%A0%E9%99%90%E8%AF%95%E7%94%A8%E6%9C%80%E9%AB%98%E7%94%BB%E8%B4%A8
# https://greasyfork.org/zh-CN/scripts/467474-bilibili-%E9%98%B2%E6%AD%A2%E8%A7%86%E9%A2%91%E8%A2%AB%E8%87%AA%E5%8A%A8%E6%9A%82%E5%81%9C%E5%8F%8A%E5%BC%B9%E5%87%BA%E7%99%BB%E5%BD%95%E7%AA%97%E5%8F%A3
# https://greasyfork.org/zh-CN/scripts/470714-bilibili-b%E7%AB%99-%E6%9C%AA%E7%99%BB%E5%BD%95%E8%B4%A6%E5%8F%B7%E5%8F%AF%E4%BB%A5%E4%BD%BF%E7%94%A8%E6%9C%80%E9%AB%98%E7%94%BB%E8%B4%A8
# https://greasyfork.org/zh-CN/scripts/473498-bilibili-%E5%9C%A8%E6%9C%AA%E7%99%BB%E5%BD%95%E7%9A%84%E6%83%85%E5%86%B5%E4%B8%8B%E7%85%A7%E5%B8%B8%E5%8A%A0%E8%BD%BD%E8%AF%84%E8%AE%BA

# 用 d 模式创建页面对象（默认模式）
page = ChromiumPage()
# page = ChromiumPage(addr_driver_opts=co)
# 跳转到登录页面
page.get("https://www.bilibili.com/video/BV1J24y1c7kE") # and then it will continue execution

# looks like it is the policy. let's turn it off.
# will fail if there is no such button
page.ele('点击恢复音量').click() # working.
# page.ele('@value=点击恢复音量').click()

# span
SAVE_PATH = r"F:\WebpageScreencast"
VIDEO_NAME = "bilibili_spinning_cat.mp4"
# now let's get screenshot.
# page.screencast.start(save_path = SAVE_PATH) # has audio, but is glitchy
# # we are just watching video. no actions involved.
# import time
# time.sleep(15)
# page.screencast.stop(VIDEO_NAME)
# page.quit()

# page.quit() # unless you quit
# page.get('https://gitee.com/login')

# # 定位到账号文本框并输入账号
# page.ele('#user_login').input('您的账号')
# # 定位到密码文本框并输入密码
# page.ele('#user_password').input('您的密码')

# # 点击登录按钮
# page.ele('@value=登 录').click()
