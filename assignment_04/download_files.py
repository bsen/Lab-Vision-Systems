"""A script to automatically download images from the duckduckgo search
"""

# code of the search_images_ddg function
# is from https://github.com/fastai/fastbook/blob/master/utils.py

from fastai.vision.all import *
from nbdev.showdoc import *
import requests

FOLDER='robots'
BASE_NAME='robot'
SEARCH_WORD='robot'
N_IMAGES=10

def search_images_ddg(key,max_n=200):
     """Search for 'key' with DuckDuckGo and return a unique urls of 'max_n' images
        (Adopted from https://github.com/deepanprabhu/duckduckgo-images-api)
     """
     url        = 'https://duckduckgo.com/'
     params     = {'q':key}
     res        = requests.post(url,data=params)
     searchObj  = re.search(r'vqd=([\d-]+)\&',res.text)
     if not searchObj: print('Token Parsing Failed !'); return
     requestUrl = url + 'i.js'
     headers    = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:71.0) Gecko/20100101 Firefox/71.0'}
     params     = (('l','us-en'),('o','json'),('q',key),('vqd',searchObj.group(1)),('f',',,,'),('p','1'),('v7exp','a'))
     urls       = []
     while True:
         try:
             res  = requests.get(requestUrl,headers=headers,params=params)
             data = json.loads(res.text)
             for obj in data['results']:
                 urls.append(obj['image'])
                 max_n = max_n - 1
                 if max_n < 1: return L(set(urls))     # dedupe
             if 'next' not in data: return L(set(urls))
             requestUrl = url + data['next']
         except:
             pass

urls = search_images_ddg(SEARCH_WORD, max_n=N_IMAGES)

print(len(urls), urls[0])

for i, url in enumerate(urls):
    print(f'Downloading {url}')
    r = requests.get(url)

    file_extension = url.split('?')[0].split('.')[-1]

    filename = f'{FOLDER}/{BASE_NAME}{i}.{file_extension}'

    with open(filename, 'wb') as file:
        file.write(r.content)

