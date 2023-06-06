import itertools
import os.path
import pickle

from tqdm import tqdm

from nnsummary.config import ASPECT_MINING_MINIMUM_TEXTS, ASPECT_MINING_MINIMUM_SUPPORT, FILE_CACHE_ASPECT_MINING, \
    MIN_ASPECTS_TO_TRAIN, CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH
from nnsummary.wikipedia.category_tree import CategoryTree
from nnsummary.wikipedia.people_processor_clustering import WikipediaPeopleProcessorWithClustering


class AspectMining:

    def __init__(self):

        if os.path.isfile(FILE_CACHE_ASPECT_MINING):
            print(f'Loading aspect mining results from {FILE_CACHE_ASPECT_MINING}')
            self.wiki = WikipediaPeopleProcessorWithClustering()
            self.cat_tree = CategoryTree()
            with open(FILE_CACHE_ASPECT_MINING, 'rb') as f:
                self.mining_results = pickle.load(f)
        else:
            self.wiki = WikipediaPeopleProcessorWithClustering()
            self.cat_tree = CategoryTree()
            self.mining_results = self.mine_aspect_candidates()
            print(f'Writing aspect mining results to {FILE_CACHE_ASPECT_MINING}')
            with open(FILE_CACHE_ASPECT_MINING, 'wb') as f:
                pickle.dump(self.mining_results, f)

    def get_person_aspects_for_class(self, person_class):
        aspects = set()
        for entry in self.mining_results:
            if entry[0] == person_class:
                aspects.add(entry[1])
        aspects = list(aspects)
        aspects.sort()
        return aspects

    def get_relevant_person_classes(self):
        distinct_person_classes = {a for a in self.get_distinct_person_classes()
                                   if len(self.get_person_aspects_for_class(a)) >= MIN_ASPECTS_TO_TRAIN}
        distinct_person_classes_lvx = self.cat_tree.filter_occupations_by_level(distinct_person_classes,
                                                                                CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH)
        print(f'Level {CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH}-classes that have more than two aspects: '
              f'{len(distinct_person_classes_lvx)}')
        return sorted(distinct_person_classes_lvx)

    def get_distinct_person_classes(self):
        return set([r[0] for r in self.mining_results])

    def mine_aspect_candidates(self):
        print('Begin aspect mining...')
        relevant_titles = {t for t in self.wiki.section2count if
                           self.wiki.section2count[t] >= ASPECT_MINING_MINIMUM_TEXTS}
        relevant_categories = {c for c in self.wiki.wiki.all_used_categories if
                               len(self.wiki.wiki.category2person[c]) >= ASPECT_MINING_MINIMUM_TEXTS}

        cat_title_combinations = list(itertools.product(relevant_categories, relevant_titles))
        print(f'Need to check {len(cat_title_combinations)} combinations...')

        mining_results = []
        for category, title in tqdm(cat_title_combinations):
            abs_sup = self.support(title, category)
            sup = self.rel_support(title, category)
            # if sup > MIN_SUPPORT:
            if sup >= ASPECT_MINING_MINIMUM_SUPPORT and abs_sup >= ASPECT_MINING_MINIMUM_TEXTS:
                spec = self.specificity(title, category)
                mining_results.append((category, title, sup, spec, abs_sup))
        return mining_results

    def support(self, aspect, category):
        if category not in self.wiki.wiki.category2person:
            return 0

        return len([p for p in self.wiki.wiki.category2person[category] if aspect in self.wiki.person2titles[p]])

    def rel_support(self, aspect, category):
        if category not in self.wiki.wiki.category2person:
            return 0

        return self.support(aspect, category) / len(self.wiki.wiki.category2person[category])

    def specificity(self, aspect, category):
        if category not in self.wiki.wiki.category2person:
            return 0

        # Has the category super categories
        super_cs = self.cat_tree.find_super_categories(category)
        if len(super_cs) == 0:
            return 1.0

        cs_persons = set()
        for super_c in super_cs:
            cs_persons.update(
                [p for p in self.wiki.wiki.category2person[super_c] if aspect in self.wiki.person2titles[p]])

        return self.support(aspect, category) / len(cs_persons)
