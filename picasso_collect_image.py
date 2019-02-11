'''
This contains functions to collect images of Picasso's paintings from
wikiart.org. 
'''

from bs4 import BeautifulSoup
import requests
import re
from time import sleep
import os
import urllib

#specify the path in which the images will be saved
path = '/Users/Alice/Programs/Python/Picasso/'

def make_decade_folders(path):
    '''
    Creates a Year folder, and subfolders of each decade from 1890 to 1970.
    path is the directory in which the folders will be created
    '''
    
    #list of the names of each subfolder
    decades = ['1890', '1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970']
    #path of the Year folder that will be created
    folder_path = path + 'Year'

    #Create the Year folder
    try:  
        os.mkdir(folder_path)
    except OSError:  
        print ("Creation of the directory %s failed" % folder_path)
    else:  
        print ("Successfully created the directory %s " % folder_path)
    
    #Within the Year, create the subfolders
    for decade in decades:
        try:  
            os.mkdir(folder_path + '/' + decade)
        except OSError:  
            print ("Creation of the subdirectory %s failed" % decade)
        else:  
            print ("Successfully created the subdirectory %s " % decade)

def save_image_by_year(url, path):
    '''
    Downloads the images of Picasso's painting listed in wikiart,
    and save it to the appropriate subfolder. In order to use this function,
    you must have the folders (Year folder and the decade subfolders within it)
    already created. If the folders have not been created, run the
    make_decades_folders() first.
    url is the link to the image
    path is the path of the Year folder
    '''
    
    #path to subfolders for each decade
    folder_1890 = path + '1890/'
    folder_1900 = path + '1900/'
    folder_1910 = path + '1910/'
    folder_1920 = path + '1920/'
    folder_1930 = path + '1930/'
    folder_1940 = path + '1940/'
    folder_1950 = path + '1950/'
    folder_1960 = path + '1960/'
    folder_1970 = path + '1970/'
    
    #navigate to painting info
    painting_soup = BeautifulSoup(requests.get(url).text, 'html5lib')
    painting_info = painting_soup.find('div', {'class': 'wiki-container'}).find(
                                       'div', {'class': 'wiki-container-responsive'}).find(
                                       'section', {'class': 'wiki-layout-left-menu'}).find(
                                       'main', {'ng-controller': 'ArtworkViewCtrl'}).find(
                                       'div', {'class': 'wiki-layout-artist-info'})
    #extract information for the year painted
    raw_year = painting_info.find('article').find('ul').find('li').find(
                                  'span', {'itemprop': 'dateCreated'})
    if raw_year is None:
        li = painting_info.find('article').find('ul').find_all('li')
        for i, info in enumerate(li):
            if "Date" in info.text:
                raw_year = li[i].find('span', {'itemprop': 'dateCreated'})
    if raw_year is not None:
        year = int(raw_year.text)
    #if year is unknown, assign year as None
    else:
        year = None

    #extract the url of the image     
    download_url = painting_info.find('aside').find(
                                      'div', {'ng-controller': 'ArrowsCtrl'}).find(
                                      'img', {'itemprop': 'image'}).get('src')
    #convert url to ASCII characters
    download_url = urllib.quote(download_url.encode('utf8'), ':/')
    url = urllib.quote(url.encode('utf8'), ':/')
    #assign a name for the image
    image_name = url.split('/')[5] + '.jpg'
    
    #save the image to the appropriate decade folder
    #if image has no year, do not save
    if year < 1900:
        urllib.urlretrieve(download_url, folder_1890+image_name)
    elif year < 1910:
        urllib.urlretrieve(download_url, folder_1900+image_name)
    elif year < 1920:
        urllib.urlretrieve(download_url, folder_1910+image_name)
    elif year < 1930:
        urllib.urlretrieve(download_url, folder_1920+image_name)
    elif year < 1940:
        urllib.urlretrieve(download_url, folder_1930+image_name)
    elif year < 1950:
        urllib.urlretrieve(download_url, folder_1940+image_name)
    elif year < 1960:
        urllib.urlretrieve(download_url, folder_1950+image_name)
    elif year < 1970:
        urllib.urlretrieve(download_url, folder_1960+image_name)
    elif year < 1980:
        urllib.urlretrieve(download_url, folder_1970+image_name)
    else:
        print "Image " + url + "was not saved because painting has no date."
    
#Use the function below if making folders for the first time
#make_decade_folders(path)
image_by_year_path = '/Users/Alice/Programs/Python/Picasso/Year/'
image_by_period_path = '/Users/Alice/Programs/Python/Picasso/Period/'

#link to the list of all the paintings 
url = 'https://www.wikiart.org/en/pablo-picasso/all-works/text-list'
soup = BeautifulSoup(requests.get(url).text, 'html5lib')

#navigate to the list of paitings
paintings = soup.find('div', {'class': 'wiki-container'}).find(
                          'div', {'class': 'wiki-container-responsive'}).find(
                          'section', {'class': 'wiki-layout-left-menu'}).find(
                          'main', {'class': 'view-all-works-main'}).find(
                          'ul', {'class': 'painting-list-text'}).find_all(
                          'li', {'class': 'painting-list-text-row'})

#for each painting link, save the image
for painting in paintings:    
    #obtain the url of the painting
    painting_url = 'https://www.wikiart.org' + painting.find('a').get('href')
    save_image_by_year(painting_url, image_by_year_path)



