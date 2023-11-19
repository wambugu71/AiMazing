import requests
import os
import json

os.environ["API_KEY"] = "K89635879588957"

def ocr_space_file(filename, overlay=False, language='eng'):


    payload = {'isOverlayRequired': overlay,
               'apikey': os.environ["API_KEY"],
               'language': "eng",
               }
    with open(filename, 'rb') as f:
        response = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return response

#text = ocr_space_file(filename='test1.jpg', language='eng')
#results = ocr_space_file(filename='test3.jpg', language='eng').json()
#results = text["ParsedResults"][0]["ParsedText"]
#res1 = results['ParsedResults'][0]['ParsedText']
#[0]["ParsedText"])
#print(res1)