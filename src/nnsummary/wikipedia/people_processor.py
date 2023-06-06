import json
import os
import pickle
from collections import defaultdict

from tqdm import tqdm

from nnsummary.config import PATH_WIKIPEDIA_JSONS, SECTION_MIN_LENGTH, SECTIONS_TO_IGNORE, \
    FILE_CACHE_WIKIPEDIA_PEOPLE_DATA, WIKIPEDIA_MIN_SUMMARY_LENGTH, WIKIPEDIA_IN_SECTIONS
from nnsummary.wikipedia.category_tree import clean_category_name, CategoryTree


def clean_section_title(title):
    t = title.lower()
    t = t.replace(' (selectie)', '')
    t = f' {t} '.replace(' als ', '')
    return t.strip()


class WikipediaPeopleProcessor:

    def __init__(self):
        self.category2count = {}
        self.section2count = {}
        self.section2texts = {}

        self.persons = set()
        self.category2person = {}
        self.person2categories = {}
        self.person2titles = {}
        self.person_title2text = {}

        self.all_section_titles = set()
        self.all_used_categories = set()

        if os.path.isfile(FILE_CACHE_WIKIPEDIA_PEOPLE_DATA):
            print(f'Load people data from cache {FILE_CACHE_WIKIPEDIA_PEOPLE_DATA}...')
            with open(FILE_CACHE_WIKIPEDIA_PEOPLE_DATA, 'rb') as f:
                self.__dict__ = pickle.load(f)
        else:
            print('Create new people data...')
            self.__process_wikipedia_people_json()
            print(f'Store people data to cache: {FILE_CACHE_WIKIPEDIA_PEOPLE_DATA}')
            with open(FILE_CACHE_WIKIPEDIA_PEOPLE_DATA, 'wb') as f:
                pickle.dump(self.__dict__, f)

    def __process_wikipedia_people_json(self):
        cat_tree = CategoryTree()

        peoples = []
        for filename in os.listdir(PATH_WIKIPEDIA_JSONS):
            if filename.lower().endswith(".json"):
                peoples.append(os.path.join(PATH_WIKIPEDIA_JSONS, filename))

        print(len(peoples))
        print(peoples[0])

        duplicated_section, skipped, processed = 0, 0, 0
        for path in tqdm(peoples):
            with open(path, 'rt') as f:
                data = json.load(f)

                titles = set()
                # get the section texts
                for sec in data["section"]:
                    if len(sec["text"].strip()) < SECTION_MIN_LENGTH:
                        continue
                    title = clean_section_title(sec["title"])
                    titles.add(title)

                titles = titles - SECTIONS_TO_IGNORE

                if len(data["summary"]) < WIKIPEDIA_MIN_SUMMARY_LENGTH:
                    skipped += 1
                    continue
                if len(titles) < WIKIPEDIA_IN_SECTIONS:
                    skipped += 1
                    continue

                # Anaylze occupations
                categories = set()
                for cat in data["occupations"]:
                    cat = clean_category_name(cat)
                    if cat_tree.is_known(cat):
                        categories.add(cat)
                        categories.update(cat_tree.find_super_categories(cat))

                # Did we find a category?
                if len(categories) == 0:
                    skipped += 1
                    continue

                # Filter and only keep categories to a certain level
                # categories = cat_tree.filter_occupations_by_level(categories, MAX_CATEGORY_LEVEL)

                # Add person to set
                person = data["title"].lower().strip()
                self.persons.add(person)
                self.person2categories[person] = categories

                titles = set()
                # get the section texts
                for sec in data["section"]:
                    text = sec["text"].strip().replace('\n', '')
                    if len(text) < SECTION_MIN_LENGTH:
                        continue
                    title = clean_section_title(sec["title"])
                    if title in SECTIONS_TO_IGNORE:
                        continue

                    titles.add(title)

                    key = (person, title)
                    if key not in self.person_title2text:
                        self.person_title2text[key] = text
                    else:
                        self.person_title2text[key] += ' ' + text
                        duplicated_section += 1
                        # raise ValueError(f'Duplicated Section for person {person}')

                    if title not in self.section2texts:
                        self.section2texts[title] = [text]
                    else:
                        self.section2texts[title].append(text)

                titles = titles - SECTIONS_TO_IGNORE
                for t in titles:
                    if t not in self.section2count:
                        self.section2count[t] = 1
                    else:
                        self.section2count[t] += 1
                self.all_section_titles.update(titles)

                # Add person2title mapping
                self.person2titles[person] = titles

                # Now store to which categories the person belongs to
                for c in categories:
                    self.all_used_categories.add(c)
                    if c not in self.category2person:
                        self.category2person[c] = {person}
                    else:
                        self.category2person[c].add(person)

                # Begin article processing
                processed += 1
                # Count categories
                for c in categories:
                    if c not in self.category2count:
                        self.category2count[c] = 1
                    else:
                        self.category2count[c] += 1

        print(f'{skipped} articles skipped / {processed} articles processed')
        print(f'{duplicated_section} duplicated section titles found')

    def print_statistics(self):
        peoples = []
        for filename in os.listdir(PATH_WIKIPEDIA_JSONS):
            if filename.lower().endswith(".json"):
                peoples.append(os.path.join(PATH_WIKIPEDIA_JSONS, filename))

        print(f'Before filtering: {len(peoples)} Wikipedia articles')
        print(f'After filtering: {len(self.person2titles)} Wikipedia articles')


def main() -> int:
    people_processor = WikipediaPeopleProcessor()
    people_processor.print_statistics()
    return 0


if __name__ == '__main__':
    main()
