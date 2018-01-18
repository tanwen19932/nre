import urllib, sys,urllib.request

def getWeather():
    host = u'http://jisutqybmf.market.alicloudapi.com'
    path = u'/weather/query'
    method = 'GET'
    appcode = u'e285d97f69774482a37253a85384dc5a'
    querys = 'city=%E5%AE%89%E9%A1%BA&citycode=citycode&cityid=cityid&ip=ip&location=location'
    bodys = {}
    url = host + path + '?' + querys

    request = urllib.request.Request(url)
    request.add_header('Authorization', 'APPCODE ' + appcode)
    response = urllib.request.urlopen(request)
    content = response.read().decode("UTF-8")
    if (content):
       return content
    else:
        return None


print(getWeather())