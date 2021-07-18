import os
import pandas as pd
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup as bs

df = pd.read_csv('EVALUATION/evaluation_poi_links - Sheet1.csv')

firefox_options = Options()
firefox_options.headless = True
driver = Firefox(options = firefox_options)

count = 1

for link in df.iloc[:, 0]:
    print(count)
    count += 1
    name = str(str(link).split('.html')[0]).split('-')[-2] + str(str(link).split('.html')[0]).split('-')[-1]
    if name + ".csv" in os.listdir('EVALUATION/popular_mentions'):
        continue

    try:
        driver.get(link)
        source = bs(driver.page_source, 'html.parser')
        elements = source.find('div', attrs = {'class' : '_3oYukhTK'}).find_all('button')
        text = "The popular mentions are :-"
        for element in elements:
            text += " " + element.get_text()

    except:
        continue
    
    df1 = pd.DataFrame()
    df1['populars'] = [text.split()]
    df1.to_csv('EVALUATION/popular_mentions/' + name + '.csv')

driver.close()