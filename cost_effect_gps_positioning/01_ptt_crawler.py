#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Howard
"""

import pandas as pd
from ptt import Board, crawl_ptt_page, crawl_ptt_page_auto
from requests.exceptions import ReadTimeout
import numpy as np
import time
import random 
import jieba
jieba.set_dictionary('dict.txt.big')

## 必要設定的欄位
# 1. Board_Name:放入讀者想要爬取的版名
# 2. page_num：看看想要爬取幾頁
KoreaDrama = crawl_ptt_page_auto(Board_Name ='KoreaDrama' ,
                            page_num= 1)

# 要使用utf-8-sig存檔
KoreaDrama.to_csv('KoreaDrama_test.csv',encoding = 'utf-8-sig') #存檔
