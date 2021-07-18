import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
from selenium import webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
from googlesearch import search

def Sel_Ext(Destination):
  revi = []
  query = (" tripadvisor reviews about " + str(Destination))
  for j in (search(query,tld='com',lang='en',num=4)):
    outputs = str(j)
    print(outputs)
    if "tripadvisor" and "com" in outputs:
      url = j
      break
  wd = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
  print(str(url))
  wd.get(str(url))
  l=0
  with open("Rev.csv","w") as csvfile:
    filewriter=csv.writer(csvfile,delimiter=",")
    for j in range(0,12):
      for t in range(0,1):
        te=wd.find_elements_by_class_name("cPQsENeY")
        l=+1
        try:
          for i in (te):             
            p=i.text
#            print(p)
            revi.append(p)
            filewriter.writerow([p])  
        except:
          wd.implicitly_wait(40)
          continue
          
    #l=str(i*2)  
  #    print(j)
  #    if(j!=0):
  #      button=wd.find_element_by_xpath("//*[@id='component_19']/div[3]/div/div[8]/div/a[2]")
  #    else:
  #    button=wd.find_element_by_xpath("//*[@id='spnPaging']/li["+str(int(j+1))+"]/a") 
      gu = j+1
      if gu<=4:
        button=wd.find_element_by_xpath("//*[@id='component_20']/div[3]/div/div[8]/div/div/a["+str(int(gu))+"]")
  #    else:                           //*[@id="ctl00_ctl00_ContentPlaceHolderFooter_ContentPlaceHolderBody_litPages"]/ul/li[3]/a
  #      button = wd.find_element_by_xpath("//*[@id='spnPaging']/li[2]/a")
      else:
        button=wd.find_element_by_xpath("//*[@id='component_20']/div[3]/div/div[8]/div/div/a["+str(4)+"]")  
      if gu<=4:
        WebDriverWait(wd, 20.05).until(EC.element_to_be_clickable((By.XPATH,"//*[@id='component_20']/div[3]/div/div[8]/div/div/a["+str(int(gu))+"]"))).click()
      else:                                                                  
        WebDriverWait(wd, 20.05).until(EC.element_to_be_clickable((By.XPATH,"//*[@id='component_20']/div[3]/div/div[8]/div/div/a["+str(int(4))+"]"))).click()
      time.sleep(1)
      print("1")
  return(revi)

