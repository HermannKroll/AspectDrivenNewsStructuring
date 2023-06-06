from nnsummary.config import WIKIPEDIA_OCCUPATIONS_PATH
import os
import requests
import json

category = "Categorie:Persoon naar beroep"


def get_subcategories(category):
    url = "https://nl.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max",
        "cmtype": "subcat"
    }
    subcategories_list = []
    while True:
        response = requests.get(url, params=params).json()
        subcategories_list += response["query"]["categorymembers"]
        if "continue" not in response:
            break
        params["cmcontinue"] = response["continue"]["cmcontinue"]
    return subcategories_list


def crawl_subcategories(category, visited=set()):
    subcategories_list = get_subcategories(category)
    visited.add(category)
    subcategories_dict = {}
    for subcategory in subcategories_list:
        subcategory_title = subcategory["title"]
        if subcategory_title not in visited:
            subcategory_name = subcategory_title[10:]
            print(subcategory_name)
            subcategories_dict[subcategory_name] = crawl_subcategories(subcategory_title, visited)
    return subcategories_dict


categories_dict = crawl_subcategories(category)

with open(WIKIPEDIA_OCCUPATIONS_PATH, 'w') as outfile:
    json.dump(categories_dict, outfile, indent=2)
