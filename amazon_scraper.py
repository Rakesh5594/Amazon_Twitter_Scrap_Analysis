import requests
from bs4 import BeautifulSoup

def amazon(url,n):
    all_reviews = []
    # agent spoofing, Amazon doesn't allow boots scrap, to fool the server we can say as logging through a website.
    HEADERS = {'user-agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5)'
                              'AppleWebKit/537.36 (KHTML, like Gecko)'
                              'Chrome/45.0.2454.101 Safari/537.36')}
    pgnum = url.index("&pageNumber=") + len("&pageNumber=")
    for p in range(n):
        print("Page " + str(p + 1) + "/" + str(n))
        each_url = url[0:pgnum]+str(p+1)
        page = requests.get(each_url, headers=HEADERS)
        page_text = page.text
        soup = BeautifulSoup(page_text, 'html5lib')
        each_pg_revs = soup.find_all('span', {'class': 'a-size-base review-text review-text-content'})
        for each_rev in each_pg_revs:
            all_reviews.append(each_rev.text)

    return all_reviews





