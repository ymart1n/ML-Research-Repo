import re
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import pprint

def get_moive_data(movie_url):#爬取单部电影的数据
    url = "http://www.imdb.com" + movie_url
    page_response = requests.request("GET", url)
    if page_response.status_code == 200:
       movie_soup = BeautifulSoup(page_response.content,'html.parser')
       
       name_temp = movie_soup.select_one('.title_wrapper').select_one('h1').get_text()
       name = name_temp.replace('\xa0', '')
       
       id_pattern = re.compile(r'(?<=tt)\d+(?=/?)')
       movie_id = int(id_pattern.search(movie_url).group())
       
       score_title = movie_soup.select_one('span[itemprop="ratingValue"]')
       if (score_title != None):
           score = float(score_title.get_text())
           released = True
       else:
           score = None
           released = False
           
       ratingCount_title = movie_soup.select_one('span[itemprop="ratingCount"]')
       if (ratingCount_title != None):
           ratingCount_string = ratingCount_title.get_text()
           ratingCount = int(ratingCount_string.replace(',', ''))
       else:
           ratingCount = None
           
       year_title = movie_soup.select_one('span[id="titleYear"]')
       if (year_title != None): 
           year = int(year_title.select_one('a').get_text())
       else:
           year = None
           
       director_title = movie_soup.select_one('.credit_summary_item')    
       if (director_title != None):
           director = director_title.select_one('a').get_text()
       else:
           director = None
    
       movie_data = {
        'ID':movie_id, #电影ID号，int
        'Name': name, #电影名称，string
        'Released': released, #是否上映，bool
        'Year':year, #上映年份，int，若未上映则返回None
        'Score':score, #评分，float，如无评分则返回None
        'RatingCount':ratingCount,    #评价人数，int，若无人数则返回None
        'Director':director  #导演，string，若无导演则返回None
        }     
       return movie_data
   
    else:
       print("Error when request URL")
       return None 


def parse_page(base_url, page_url):#爬取每页的50部电影数据
    url = f'{base_url}{page_url}'
    response = requests.request("GET",url)
    print(response)
    page_soup = BeautifulSoup(response.content,'html.parser')
    links = page_soup.select('.lister-item-content')
    page_pagination = page_soup.select_one('.desc')
    #print(page_pagination)
    page_pagination_links = page_pagination.select('a')
    next_page = ""
    for page in page_pagination_links:
        if page.text == "Next »":
            next_page = page['href']
    for link in links:
        movie_link = link.select_one('.lister-item-header').select_one('a')['href']
        
        data = get_moive_data(movie_link)
        pprint.pprint(data)
        ##这里依次输出每部电影的数据，具体格式见movie_data
    return next_page     
        
def main():
    base_url = "https://www.imdb.com"
    page_url = "/search/title/?genres=comedy"
    for i in range(10000):#爬取10000页即50w部电影数据，可自行修改
        page_url = parse_page(base_url, page_url)

        
  
if __name__ == '__main__':
    main()