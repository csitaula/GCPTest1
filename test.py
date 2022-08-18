import requests
req=requests.post("http://127.0.0.1:5000/",files={'file':open('aug_0_2751.png','rb')})
print(req)
print(req.json())