import re
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import pprint


def from_dict_to_movie(the_movie_dict):
    return Movie(movie_id=the_movie_dict['ID'],
                 name=the_movie_dict['Name'],
                 released=the_movie_dict['Released'],
                 year=the_movie_dict['Year'],
                 score=the_movie_dict['Score'],
                 rating_count=the_movie_dict['RatingCount'],
                 director=the_movie_dict['Director']
                )



def get_moive_data(movie_url): # This function scrap the data of one single movie
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
        'ID':movie_id, # the id of movie in IMDb，int
        'Name': name, # the name of movie，string
        'Released': released, # movie is released or not，bool
        'Year':year, # year of release，int，None if it has not been released 
        'Score':score, # the rating score of movie，float，None if it has not been rated
        'RatingCount':ratingCount, # the numebr of ratings，int，None if no rating
        'Director':director # the name of director，string，None if no director
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
        try:
            data = get_moive_data(movie_link)
            from_dict_to_movie(data).save()
        except requests.exceptions.RequestException as e:
            print(e)
            continue
    return next_page     
        
def main():
    base_url = "https://www.imdb.com"
    page_url = "/search/title/?genres=comedy"
    socket.setdefaulttimeout(20)
    for i in range(10000): # Scaping 10,000 pages ~ 500,000 movies
        try:
            page_url = parse_page(base_url, page_url)
        except requests.exceptions.RequestException as e:
            print(e)
            continue

        
  
if __name__ == '__main__':
    main()