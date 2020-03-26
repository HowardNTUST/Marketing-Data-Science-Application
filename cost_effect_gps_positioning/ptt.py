# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:55:23 2019

@author: Howie
"""
from requests.exceptions import ReadTimeout
import time

import pandas as pd
import os
import urllib
import datetime
import collections
import json
import argparse

import requests
from bs4 import BeautifulSoup

# error control


# 寫成function    
def ptt_crawl_one(Board_Name):
        
    # 抓該板首頁的文章
    latest_page = Board(Board_Name)
    
    # 抓取資料
    content=[] #文章內容
    titles=[] #文章標題
    times=[] #文章時間
    for summary in latest_page: # 只要抓最新的頁面
        article = ptt_error_handling(summary)
        print('正在抓資料中...'+summary.title)
        if article !='':        
            # 將所有內容儲存在一個[]
            titles.append(article.title)
            content.append(article.content)
            times.append(str(article.datetime))
        
        
    # 將結果做成df
    dic = {'標題':titles,
           '時間':times,
           '內容':content
           }
    
    
    final_data = pd.DataFrame(dic)
    
    # 如何去除空白的標題
    final_data = final_data[final_data['標題'] !='']

    return final_data



# 寫成function
def crawl_ptt_page(Board_Name,start,page_num):
    listt = []
    for i in range(page_num):
        my_crawl_list = ptt_crawl(Board_Name= Board_Name, start=start , page = i)
        listt.append(my_crawl_list)
    listtdf = pd.concat(listt)
    return listtdf 


def ptt_crawl_v2(Board_Name, start, page):
        
    # 抓該板首頁的文章
    latest_page = Board(Board_Name, start-page)
    
    # 抓取資料
    content=[] #文章內容
    titles=[] #文章標題
    times=[] #文章時間
    for summary in latest_page: # 只要抓最新的頁面

        article = ptt_error_handling(summary)
        print('正在抓資料中...'+summary.title)
        
        # 將所有內容儲存在一個[]
        if article !='':
            titles.append(article.title)
            content.append(article.content)
            times.append(str(article.datetime))
        
        
    # 將結果做成df
    dic = {'標題':titles,
           '時間':times,
           '內容':content
           }
    
    
    final_data = pd.DataFrame(dic)
    
    # 問題：如何去除空白的標題
    final_data = final_data[final_data['標題'] !='']

    return final_data



def ptt_error_handling(summary):
    try:
        article = summary.read()
        return article 
    except:
        return ''
    

# exception
class Error(Exception):
    """Base class for all exceptions raised by this module"""
    pass


class InValidBeautifulSoupTag(Error):
    """Can not create ArticleSummary because of invalid bs tag"""
    pass


class NoGivenURLForPage(Error):
    """Given None or empty url when build page"""
    pass


class PageNotFound(Error):
    """Can not fetch page by given url"""
    pass


class ArtitcleIsRemoved(Error):
    """Can not read removed article from ArticleSummary"""
    pass


# utility
def parse_std_url(url):
    """Parse standard ptt url
    >>> parse_std_url('https://www.ptt.cc/bbs/Gossiping/M.1512057611.A.16B.html')
    ('https://www.ptt.cc/bbs', 'Gossiping', 'M.1512057611.A.16B')
    """
    prefix, _,  basename = url.rpartition('/')
    basename, _, _ = basename.rpartition('.')
    bbs, _, board = prefix.rpartition('/')
    bbs = bbs[1:]
    return bbs, board, basename


def parse_title(title):
    """Parse article title to get more info
    >>> parse_title('Re: [問卦] 睡覺到底可不可以穿襪子')
    ('問卦', True, False)
    """
    _, _, remain = title.partition('[')
    category, _, remain = remain.rpartition(']')
    category = category if category else None
    isreply = True if 'Re:' in title else False
    isforward = True if 'Fw:' in title else False
    return category, isreply, isforward


def parse_username(full_name):
    """Parse user name to get its user account and nickname
    >>> parse_username('seabox (歐陽盒盒)')
    ('seabox', '歐陽盒盒')
    """
    name, nickname = full_name.split(' (')
    nickname = nickname.rstrip(')')
    return name, nickname


# Msg is a namedtuple which used to model the info of one of the pushes
Msg = collections.namedtuple('Msg', ['type', 'user', 'content', 'ipdatetime'])


class ArticleSummary:
    """Class used to model the article info in ArticleListPage"""

    def __init__(self, title, url, score, date, author, mark, removeinfo):
        # title
        self.title = title
        self.category, self.isreply, self.isforward = parse_title(title)

        # url
        self.url = url
        _, self.board, self.aid = parse_std_url(url)

        # meta
        self.score = score
        self.date = date
        self.author = author
        self.mark = mark

        # remove
        self.isremoved = True if removeinfo else False
        self.removeinfo = removeinfo

    @classmethod
    def from_bs_tag(cls, tag):
        """classmethod for create a ArticleSummary object from corresponding bs tag"""
        try:
            removeinfo = None
            title_tag = tag.find('div', class_='title')
            a_tag = title_tag.find('a')

            if not a_tag:
                removeinfo = title_tag.get_text().strip()

            if not removeinfo: 
                title = a_tag.get_text().strip()
                url = a_tag.get('href').strip()
                score = tag.find('div', class_='nrec').get_text().strip()
            else:
                title = '本文章已被刪除'
                url = ''
                score = ''

            date = tag.find('div', class_='date').get_text().strip()
            author = tag.find('div', class_='author').get_text().strip()
            mark = tag.find('div', class_='mark').get_text().strip()
        except Exception:
            raise InValidBeautifulSoupTag(tag)

        return cls(title, url, score, date, author, mark, removeinfo)

    def __repr__(self):
        return '<Summary of Article("{}")>'.format(self.url)

    def __str__(self):
        return self.title

    def read(self):
        """Read the Article from url and return ArticlePage
        raise ArticleIsRemoved error if it is removed
        """
        if self.isremoved:
            raise ArtitcleIsRemoved(self.removeinfo)
        return ArticlePage(self.url)


class Page:
    """Base class of page
    fetch the web page html content by url
    all its subclass object should call its __init__ first
    """
    ptt_domain = 'https://www.ptt.cc'

    def __init__(self, url):
        if not url:
            raise NoGivenURLForPage

        self.url = url

        url = urllib.parse.urljoin(self.ptt_domain, self.url)
        resp = requests.get(url=url, cookies={'over18': '1'}, verify=True, timeout=3)

        if resp.status_code == requests.codes.ok:
            self.html = resp.text
        else:
            raise PageNotFound


class ArticleListPage(Page):
    """Class used to model article list page"""

    def __init__(self, url):
        super().__init__(url)

        # to set article_tags
        soup = BeautifulSoup(self.html, 'lxml')
        self.article_summary_tags = soup.find_all('div', 'r-ent')
        self.article_summary_tags.reverse()

        # to set related urls
        action_tags = soup.find('div', class_='action-bar').find_all('a')
        self.related_urls = {}
        url_names = 'board man oldest previous next newest'
        for idx, name in enumerate(url_names.split()):
            self.related_urls[name] = action_tags[idx].get('href')

        # to set board and idx
        _, self.board, basename = parse_std_url(url)
        _, _, idx = basename.partition('index')
        if idx:
            self.idx = int(idx)
        else:
            _, self.board, basename = parse_std_url(self.related_urls['previous'])
            _, _, idx = basename.partition('index')
            self.idx = int(idx)+1

    @classmethod
    def from_board(cls, board, index=''):
        """classmethod for create a ArticleListPage object from given board name and its index
        if index is not given, create and return the lastest ArticleListPage of the board
        """
        url = '/'.join(['/bbs', board, 'index'+str(index)+'.html'])
        return cls(url)

    def __repr__(self):
        return 'ArticleListPage("{}")'.format(self.url)

    def __iter__(self):
        return self.article_summaries

    def get_article_summary(self, index):
        return ArticleSummary.from_bs_tag(self.article_summary_tags[index])

    @property
    def article_summaries(self):
        return (ArticleSummary.from_bs_tag(tag) for tag in self.article_summary_tags)

    @property
    def previous(self):
        return ArticleListPage(self.related_urls['previous'])

    @property
    def next(self):
        return ArticleListPage(self.related_urls['next'])

    @property
    def oldest(self):
        return ArticleListPage(self.related_urls['oldest'])

    @property
    def newest(self):
        return ArticleListPage(self.related_urls['newest'])


class ArticlePage(Page):
    """class used to model article page"""

    default_attrs = ['board', 'aid', 'author', 'date', 'content', 'ip']
    default_csv_attrs = default_attrs + ['pushes.count.score']
    default_json_attrs = default_attrs + ['pushes.count', 'pushes.simple_expression']

    def __init__(self, url):
        super().__init__(url)

        _, _, self.aid = parse_std_url(url)

        # to set article_tags
        soup = BeautifulSoup(self.html, 'lxml')
        main_tag = soup.find('div', id='main-content')
        meta_name_tags = main_tag.find_all('span', class_='article-meta-tag')
        meta_value_tags = main_tag.find_all('span', class_='article-meta-value')

        # dealing meta
        try:
            self.author = meta_value_tags[0].get_text().strip()
            self.board = meta_value_tags[1].get_text().strip()
            self.title = meta_value_tags[2].get_text().strip()
            self.date = meta_value_tags[3].get_text().strip()

            self.category, self.isreply, self.isforward = parse_title(self.title)
            self.datetime = datetime.datetime.strptime(self.date, '%a %b %d %H:%M:%S %Y')
        except:
            self.author, self.board, self.title, self.date = '', '', '', ''
            self.category, self.isreply, self.isforward = '', False, False
            self.datetime = None

        # remove meta
        for tag in main_tag.select('div.article-metaline'):
            tag.extract()
        for tag in main_tag.select('div.article-metaline-right'):
            tag.extract()

        # fetch pushes and remove them
        self.pushes = Pushes(self)
        push_tags = main_tag.find_all('div', class_='push')
        for tag in push_tags:
            tag.extract()
        for tag in push_tags:
            if not tag.find('span', 'push-tag'):
                continue
            push_type = tag.find('span', class_='push-tag').string.strip(' \t\n\r')
            push_user = tag.find('span', class_='push-userid').string.strip(' \t\n\r')
            push_content = tag.find('span', class_='push-content').strings
            push_content = ' '.join(push_content)[1:].strip(' \t\n\r')
            push_ipdatetime = tag.find('span', class_='push-ipdatetime').string.strip(' \t\n\r')
            msg = Msg(type=push_type, user=push_user, content=push_content, ipdatetime=push_ipdatetime)
            self.pushes.addmsg(msg)
        self.pushes.countit()

        # handle special item
        ip_tags = main_tag.find_all('span', class_='f2')
        dic = {}
        for tag in ip_tags:
            if '※' in tag.get_text():
                key, _, value = tag.get_text().partition(':')
                key = key.strip('※').strip()
                value = value.strip()
                if '引述' in key:
                    continue
                else:
                    dic.setdefault(key, []).append(value)
                    tag.extract()
        self.ip = dic['發信站'][0].split()[-1]

        # remove richcontent
        for tag in main_tag.find_all('div', class_='richcontent'):
            tag.extract()

        # handle trans
        trans = []
        for tag in main_tag.find_all('span', class_='f2'):
            if '轉錄至看板' in tag.get_text():
                trans.append(tag.previous_element.parent)
                trans.append(tag.get_text())
                trans.append(tag.next_sibling)
                tag.previous_element.parent.extract()
                tag.next_sibling.extract()
                tag.extract()

        # split main content and signature
        self.content, self.signature = str(main_tag).split('--')[:2]
        self.content = self.content.strip()

        contents = self.content.split('\n')
        self.content = '\n'.join(content for content in contents if not ('<div' in content and 'main-content' in content))

        contents = self.signature.split('\n')
        self.signature = '\n'.join(content for content in contents if not ('</div' in content))

    @classmethod
    def from_board_aid(cls, board, aid):
        url = '/'.join(['/bbs', board, aid+'.html'])
        return cls(url)

    def __repr__(self):
        return 'ArticlePage("{}")'.format(self.url)

    def __str__(self):
        return self.title

    @classmethod
    def _recur_getattr(cls, obj, attr):
        if not '.' in attr:
            try:
                return getattr(obj, attr)
            except:
                return obj[attr]
        attr1, _, attr2 = attr.partition('.')
        obj = cls._recur_getattr(obj, attr1)
        return cls._recur_getattr(obj, attr2)

    def dump_json(self, *attrs, flat=True):
        """dump json string of this article with specified attrs"""
        data = {}
        if not attrs:
            attrs = self.default_json_attrs
        for attr in attrs:
            data[attr] = self._recur_getattr(self, attr)
        if flat:
            return json.dumps(data, ensure_ascii=False)
        else:
            return json.dumps(data, indent=4, ensure_ascii=False)

    def dump_csv(self, *attrs, delimiter=','):
        """dump csv string of this article with specified attrs"""
        cols = []
        if not attrs:
            attrs = self.default_csv_attrs
        for attr in attrs:
            cols.append(self._recur_getattr(self, attr))
        cols = [repr(col) if '\n' in str(col) else str(col) for col in cols]
        return delimiter.join(cols)


class Pushes:
    """class used to model all pushes of an article"""

    def __init__(self, article):
        self.article = article
        self.msgs = []
        self.count = 0

    def __repr__(self):
        return 'Pushes({})'.format(repr(self.article))

    def __str__(self):
        return 'Pushes of Article {}'.format(self.Article)

    def addmsg(self, msg):
        self.msgs.append(msg)

    def countit(self):
        count_types = 'all abs like boo neutral'.split()
        self.count = dict(zip(count_types, [0, 0, 0, 0, 0]))
        for msg in self.msgs:
            if msg.type == '推':
                self.count['like'] += 1
            elif msg.type == '噓':
                self.count['boo'] += 1
            else:
                self.count['neutral'] += 1

        self.count['all'] = self.count['like'] + self.count['boo'] + self.count['neutral']
        self.count['score'] = self.count['like'] - self.count['boo']

    @property
    def simple_expression(self):
        msgs = []
        attrs = ['type', 'user', 'content', 'ipdatetime']
        for msg in self.msgs:
            msgs.append(dict(zip(attrs, list(msg))))
        return msgs


# alias
Summary = ArticleSummary
Article = ArticlePage
Board = ArticleListPage.from_board


def main():
    parser = argparse.ArgumentParser(description='ptt.py')

    parser.add_argument('-b', '--board', metavar='BOARD', type=str, required=True, help='board name')
    parser.add_argument('-d', '--destination', metavar='DIR', type=str, default='', help='destination')
    parser.add_argument('-f', '--format', choices=['json', 'csv'], default='json', help='output format (default: json)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a', '--aid', metavar='ID', type=str, help='article id')
    group.add_argument('-i', '--index', metavar='START END', nargs=2, type=int, help='start/end index')

    args = parser.parse_args()

    t1 = time.time()
    if args.aid:
        article = Article.from_board_aid(args.board, args.aid)
        fname = '{}-{}.{}'.format(args.board, args.aid, args.format)
        fname = os.path.join(args.destination, fname)
        with open(fname, 'w', encoding='utf-8') as writer:
            print('dump {} to {}...'.format(article.aid, fname))
            if args.format == 'json':
                print(article.dump_json(flat=False), file=writer)
            elif args.format == 'csv':
                print(','.join(Article.default_csv_attrs), file=writer)
                print(article.dump_csv(delimiter=','), file=writer)
    else:
        latest_page_id = Board(args.board).idx
        start_idx = args.index[0] if args.index[0] >= 0 else latest_page_id + args.index[0] + 1
        end_idx = args.index[1] if args.index[1] >= 0 else latest_page_id + args.index[1] + 1
        if start_idx <= end_idx:
            fname = '{}-{}-{}.{}'.format(args.board, start_idx, end_idx, args.format)
            fname = os.path.join(args.destination, fname)
            with open(fname, 'w', encoding='utf-8') as writer:
                if args.format == 'json':
                    print('{"articles": [', file=writer)
                elif args.format == 'csv':
                    print(','.join(Article.default_csv_attrs), file=writer)
                for idx in range(start_idx, end_idx+1):
                    article_page = Board(args.board, idx)
                    for summary in article_page:
                        if summary.isremoved:
                            continue
                        article = summary.read()
                        print('dump {} to {}...'.format(article.aid, fname))
                        if args.format == 'json':
                            print(article.dump_json(flat=True), file=writer)
                        elif args.format == 'csv':
                            print(article.dump_csv(delimiter=','), file=writer)
                if args.format == 'json':
                    print(']}', file=writer)
    elapsed = time.time() - t1
    print('total in {:.3} sec.'.format(elapsed))

def ptt_crawl(Board_Name, start, page):
        
    # 抓該板首頁的文章
    latest_page = Board(Board_Name, start-page)
    
    # 抓取資料
    content=[] #文章內容
    titles=[] #文章標題
    times=[] #文章時間
    for summary in latest_page: # 只要抓最新的頁面
        if summary.isremoved:
            continue
        #睡一下，不然會被擋掉
#        time.sleep(random.randint(1,2))
        print('正在抓資料中...'+summary.title)
        
        try:
            article = summary.read()
            # 將所有內容儲存在一個[]
            titles.append(article.title)
            content.append(article.content)
            times.append(str(article.datetime))
        except:
            pass
        
        
        
    # 將結果做成df
    dic = {'標題':titles,
           '時間':times,
           '內容':content
           }
    
    
    final_data = pd.DataFrame(dic)
    
    # 問題：如何去除空白的標題
    final_data = final_data[final_data['標題'] !='']

    return final_data

def crawl_ptt_page_auto2(Board_Name, page_num):
    
    # 從自動找到最新的一頁來爬
    lst_page = ArticleListPage.from_board(Board_Name)
    start = lst_page.related_urls['previous']
    start
    
    # 從'index'這個字串進行「前後」切割
    start.split('index')
    start
    
    start = start.split('index')
    start
    
    # 問題：如何選擇list裡面的第二個元素？
    start = start[1]
    start
    
    
    # 問題：將'.html'取代成''
    start = start.replace('.html','')
    
    # 問題：將start + 1，因為start裡面的數字是前面一頁，再+1才是最新的一頁
    # https://www.ptt.cc/bbs/Gossiping/index.html
    start = int(start) +1
    start

    
    
    listt = []
    for i in range(page_num):
        listt.append(ptt_crawl(Board_Name= Board_Name, start=start , page = i))
    listtdf = pd.concat(listt)
    return listtdf 

# 寫成function
def crawl_ptt_page_auto(Board_Name, page_num):
    
    import re
    lst_page = ArticleListPage.from_board(Board_Name)
    start = lst_page.related_urls['previous']
    start =re.findall(r'\d+', start)
    start = int(start[0]) +1
    print('讓我們從最新的' + str(start ) +'開始爬取'+ str(page_num) +'頁吧'+'呵呵')
    
    listt = []
    for i in range(page_num):
        listt.append(ptt_crawl(Board_Name= Board_Name, start=start , page = i))
    listtdf = pd.concat(listt)
    return listtdf 
if __name__ == '__main__':
    main()