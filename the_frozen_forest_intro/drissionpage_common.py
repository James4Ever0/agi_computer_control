
# this thing has compatibility issue. the data is not always loaded when ported to a different device.
# maybe you need docker. we can talk about that later.

from DrissionPage.easy_set import set_paths

# this thing is not chromium-compatible
# CHROMIUM_PATH = r"C:\Users\Administrator\AppData\Local\ms-playwright\chromium-1084\chrome-win\chrome.exe"
BROWSER_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
# DOWNLOAD_PATH = r""
USER_DATA_PATH = r"F:\MicrosoftEdgeUserData"
CACHE_PATH = r"F:\MicrosoftEdgeCache"
# you may have to install extensions yourself.
USE_CHROME = False
# USE_CHROME = True
set_paths(
    browser_path=BROWSER_PATH,
    # browser_path=(CHROMIUM_PATH if USE_CHROME else BROWSER_PATH),
    # download_path=DOWNLOAD_PATH,
    # **(dict() if USE_CHROME else dict(
    user_data_path=USER_DATA_PATH,
    cache_path=CACHE_PATH #)),
)

# no sound!
# 关闭静音开播

from DrissionPage import ChromiumPage # , ChromiumOptions

# that way we may have the extensions
# co = ChromiumOptions()

import os

ext_path = "keylogger_extension/virtual-keylogger"
pathToExtension = os.path.abspath(ext_path)
pathToCORSExtension = os.path.abspath("ForceCORS")
extension_path = ",".join([pathToExtension, pathToCORSExtension])

# not very nice.
# co.set_argument('load-extension', extension_path)
# co.set_argument("disable-extensions-except", extension_path)
# co.set_argument("enable-automation")
# co.set_mute(True)
# co.set_mute(False)

# userscripts to fix bilibili ads, for babysitting
# https://greasyfork.org/zh-CN/scripts/467511-bilibili-%E5%9C%A8%E6%9C%AA%E7%99%BB%E5%BD%95%E7%9A%84%E6%83%85%E5%86%B5%E4%B8%8B%E8%87%AA%E5%8A%A8%E5%B9%B6%E6%97%A0%E9%99%90%E8%AF%95%E7%94%A8%E6%9C%80%E9%AB%98%E7%94%BB%E8%B4%A8
# https://greasyfork.org/zh-CN/scripts/467474-bilibili-%E9%98%B2%E6%AD%A2%E8%A7%86%E9%A2%91%E8%A2%AB%E8%87%AA%E5%8A%A8%E6%9A%82%E5%81%9C%E5%8F%8A%E5%BC%B9%E5%87%BA%E7%99%BB%E5%BD%95%E7%AA%97%E5%8F%A3
# https://greasyfork.org/zh-CN/scripts/470714-bilibili-b%E7%AB%99-%E6%9C%AA%E7%99%BB%E5%BD%95%E8%B4%A6%E5%8F%B7%E5%8F%AF%E4%BB%A5%E4%BD%BF%E7%94%A8%E6%9C%80%E9%AB%98%E7%94%BB%E8%B4%A8
# https://greasyfork.org/zh-CN/scripts/473498-bilibili-%E5%9C%A8%E6%9C%AA%E7%99%BB%E5%BD%95%E7%9A%84%E6%83%85%E5%86%B5%E4%B8%8B%E7%85%A7%E5%B8%B8%E5%8A%A0%E8%BD%BD%E8%AF%84%E8%AE%BA

# 用 d 模式创建页面对象（默认模式）
# page = ChromiumPage(co if USE_CHROME else None)
page = ChromiumPage()
DEFAULT_TIMEOUT = 10
page.timeout = DEFAULT_TIMEOUT
