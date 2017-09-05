from bs4 import BeautifulSoup
import requests

def findBaseEnd(origurl):
    for index in reversed(range(len(origurl))):
        if (origurl[index] == '.'):
            for index2 in range(index, len(origurl)):
                if (origurl[index2] == '/'):
                    return index2
    return len(origurl)

def completeUrl(partialUrl, base):
    if(partialUrl[0] == '/'):
        if(len(partialUrl) < 2 or partialUrl[1] != '/'):
            partialUrl = base + partialUrl
    elif(partialUrl[0] == '.'):
        index = 3
        while(partialUrl[index] == '.'):
            index = index + 3
        partialUrl = base + partialUrl[index - 1:len(partialUrl)]
    return partialUrl

def fixUrl(link):
    link = link.get('href')
    if(link != None and link != '' and link[0] != '#'):
        link = completeUrl(link,baseUrl)
        return link
    return None
    
def getLinksOnPage(url):
    try:
        r = requests.get(url)
        data = r.text
        soup = BeautifulSoup(data)
        links = soup.find_all('a')
        return links
    except:
        return None

def findEquivalentLinks(url,links):
    equivalent = []
    for link in links:
        link = fixUrl(link)
        foundLinks = getLinksOnPage(link)
        if foundLinks == links:
            equivalent.append(link)
    return equivalent


url = 'https://www.k12northstar.org/chinook'
#this is the original school webpage provided

baseUrl = url[0:findBaseEnd(url)]
#this isolates the hostname (and the protocol if present)

links = getLinksOnPage(url)
#this identifies all the hyperlinks on the original webpage

equivalentUrls = findEquivalentLinks(url,links)
equivalentUrls.append(url)
#this identifies all hyperlinks that link to a page that is identical to the original webpage

#IGNORE FOR TESTING equivalentUrls = ['https://www.k12northstar.org/Domain/35', 'https://www.k12northstar.org/Page/63', 'https://www.k12northstar.org/chinook']

headLinks = []
#

# IGNORE FOR TESTING links = ['https://www.k12northstar.org/site/Default.aspx?PageID=2678','https://www.k12northstar.org/Page/2683','http://lam.alaska.gov/sled/homework'
#         ,'https://www.k12northstar.org/Domain/36']



for link in links:
    checkForHead = [False,False]
    link = fixUrl(link)
    if (link!=None):
        foundLinks = getLinksOnPage(link)
        #print(foundLinks)
        if (foundLinks==None): continue
        for index in range(len(foundLinks)):
            foundLinks[index] = fixUrl(foundLinks[index])
            if (link == foundLinks[index]):
                checkForHead[0] = True
            if (foundLinks[index] != None):
                for equivalent in equivalentUrls:
                    if (foundLinks[index] in equivalent or equivalent in foundLinks[index]):
                        checkForHead[1] = True
        print(foundLinks)
    if checkForHead == [True,True]: headLinks.append(link)
#this loop identifies all the hyperlinks (referred to as headLinks, which is short for header links) that link back to the original webpage and link back to themselves
#these header links will likely be found in the menu/heading of the original webpage, hence the name and their importance
#however, we still have to narrow down this list

#IGNORE FOR TESTING headLinks = ['https://www.k12northstar.org/Domain/4', 'https://www.k12northstar.org/Domain/8', 'https://www.k12northstar.org/Domain/9', 'https://www.k12northstar.org/Domain/10', 'https://www.k12northstar.org/Domain/34', 'https://www.k12northstar.org/Domain/26', 'https://www.k12northstar.org/Domain/41', 'https://www.k12northstar.org/Domain/35', 'https://www.k12northstar.org/Domain/12', 'https://www.k12northstar.org/Domain/13', 'https://www.k12northstar.org/Domain/36', 'https://www.k12northstar.org/Domain/37', 'https://www.k12northstar.org/Domain/14', 'https://www.k12northstar.org/Domain/38', 'https://www.k12northstar.org/Domain/15', 'https://www.k12northstar.org/Domain/16', 'https://www.k12northstar.org/Domain/27', 'https://www.k12northstar.org/Domain/11', 'https://www.k12northstar.org/Domain/17', 'https://www.k12northstar.org/Domain/18', 'https://www.k12northstar.org/Domain/29', 'https://www.k12northstar.org/Domain/28', 'https://www.k12northstar.org/Domain/19', 'https://www.k12northstar.org/Domain/30', 'https://www.k12northstar.org/Domain/31', 'https://www.k12northstar.org/Domain/20', 'https://www.k12northstar.org/Domain/39', 'https://www.k12northstar.org/Domain/32', 'https://www.k12northstar.org/Domain/42', 'https://www.k12northstar.org/Domain/21', 'https://www.k12northstar.org/Domain/22', 'https://www.k12northstar.org/Domain/23', 'https://www.k12northstar.org/Domain/24', 'https://www.k12northstar.org/Domain/33', 'https://www.k12northstar.org/Domain/25', 'https://www.k12northstar.org/site/Default.aspx?PageType=7&SiteID=35&IgnoreRedirect=true', 'https://www.k12northstar.org/Page/63', 'https://www.k12northstar.org/site/Default.aspx?PageID=2678', 'https://www.k12northstar.org/site/Default.aspx?PageID=2678', 'https://www.k12northstar.org/site/Default.aspx?PageID=2683', 'https://www.k12northstar.org/site/Default.aspx?PageID=2684', 'https://www.k12northstar.org/site/Default.aspx?PageID=2685', 'https://www.k12northstar.org/site/Default.aspx?PageID=2686', 'https://www.k12northstar.org/site/Default.aspx?PageID=2687', 'https://www.k12northstar.org/Page/3294', 'https://www.k12northstar.org/domain/2692', 'https://www.k12northstar.org/domain/2692', 'https://www.k12northstar.org/domain/2760', 'https://www.k12northstar.org/domain/2828', 'https://www.k12northstar.org/domain/2898', 'https://www.k12northstar.org/domain/2945', 'https://www.k12northstar.org/domain/3078', 'https://www.k12northstar.org/domain/3291', 'https://www.k12northstar.org/Page/2063', 'https://www.k12northstar.org/Page/5838', 'https://www.k12northstar.org/Page/1993', 'https://www.k12northstar.org/Page/1954', 'https://www.k12northstar.org/domain/65', 'https://www.k12northstar.org/site/Default.aspx?PageType=15&SiteID=35&SectionMax=15&DirectoryType=6']

#print(headLinks)
finalLinks = []
headSet = set(headLinks)
for headLink in headLinks:
    branchLinks = getLinksOnPage(headLink)
    if (branchLinks==None): continue
    for index in range(len(branchLinks)):
        branchLinks[index] = fixUrl(branchLinks[index])
    branchLinks = set(branchLinks)
    totalLinks = headSet.union(branchLinks)
    if len(totalLinks)==len(branchLinks):
        finalLinks.append(headLink)
    else: print(len(totalLinks)-len(branchLinks))
#this loop identifies how many other headLinks each headLink links to

#Is there a way to use these numbers to determine the closed networks within the headLinks?
#How do we identify the closed network we're interested in? (the menu/heading of the webpage)

print(finalLinks)
#this will come out empty

