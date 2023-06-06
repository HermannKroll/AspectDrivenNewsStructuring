import json
import os

from tqdm import tqdm

from nnsummary.config import NEW_ARTICLE_PERSON_INFORMATION, MIN_ASPECTS_TO_TRAIN, RESULT_DIR, PATH_LM_MODEL_OUTPUT_DIR, \
    CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH, LM_CLASSIFIER_PREFIX, LM_CLASSIFIED_ARTICLE_POSTFIX

from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from nnsummary.news.articles import NewsArticleReader
from nnsummary.wikipedia.aspect_mining import AspectMining
from nnsummary.wikipedia.category_tree import CategoryTree
from nnsummary.wikipedia.people_processor_clustering import WikipediaPeopleProcessorWithClustering


class LanguageModelClassifier:

    def __init__(self, model_name, person_class, mining):
        self.name = f'{model_name}_{person_class}'
        self.person_class = person_class
        self.labels = ["no class"] + sorted(mining.get_person_aspects_for_class(person_class))
        self.path = os.path.join(f'{PATH_LM_MODEL_OUTPUT_DIR}/', f'{model_name}_{person_class}/')
        self.pipeline = None

    def __lazy_load(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.path)
        tokenizer = AutoTokenizer.from_pretrained("DTAI-KULeuven/robbert-2022-dutch-base")
        self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

    def classify_texts(self, texts: str):
        if not self.pipeline:
            self.__lazy_load()

        result_dict = self.pipeline(texts, padding=True, truncation=True)
        return [(r["label"], r["score"]) for r in result_dict]


def classify_paragraphs_of_articles(article_data, person_classes, classifiers):
    results = {}
    for c in tqdm(classifiers):
        if c.person_class not in person_classes:
            continue
        if c.name not in results:
            results[c.name] = {}

        for label in c.labels:
            if label not in results[c.name]:
                results[c.name][label] = list()

        art_content = []
        for art in article_data:
            aid = art["record_identifier"]
            contents = art["content"]
            for cont in contents:
                art_content.append((aid, cont))

        # Give everything to the pipeline
        texts = [e[1] for e in art_content]
        content_classes = c.classify_texts(texts)

        for idx, entry in enumerate(art_content):
            aid, cont = entry[0], entry[1]
            if content_classes[idx][0] != "NEGATIVE":
                rd = dict(article_id=aid, content=cont, probability=content_classes[idx][1])
                results[c.name][content_classes[idx][0]].append(rd)
    return results


def classify_articles_into_person_aspects():
    mining = AspectMining()
    wiki = mining.wiki
    cat_tree = mining.cat_tree
    articles = NewsArticleReader(verbose=False)

    classifiers = []

    person_classes = {cat_tree.root}
    for entry in NEW_ARTICLE_PERSON_INFORMATION:
        wiki_name = entry[4]
        if wiki_name in wiki.wiki.person2categories:
            person_classes.update(cat_tree.filter_occupations_by_level(wiki.wiki.person2categories[wiki_name],
                                                                       CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH))
    print(f'The following {len(person_classes)} classes are of interest')
    person_classes = sorted(person_classes)
    print(person_classes)

    model_classifier_name = LM_CLASSIFIER_PREFIX
    for pc in tqdm(person_classes):
        name = f'{model_classifier_name}'
        c = LanguageModelClassifier(name, pc, mining)
        if not os.path.isdir(c.path):
            print(f'Warning: Skipping {c.name} (path to model not available)')
            continue
        classifiers.append(c)

    for entry in NEW_ARTICLE_PERSON_INFORMATION:
        result_path = os.path.join(RESULT_DIR, f'{entry[1]}' + LM_CLASSIFIED_ARTICLE_POSTFIX)
        if os.path.isfile(result_path):
            print(f'Skipping already classified person ({result_path})')
            continue

        wiki_name = entry[4]
        if wiki_name not in wiki.wiki.person2categories:
            print(f'Person does not have categories assigned -> use {cat_tree.root}')
            person_classes = {cat_tree.root}
        else:
            person_classes = cat_tree.filter_occupations_by_level(wiki.wiki.person2categories[wiki_name],
                                                                  CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH)
        print(f'Classifying {entry[1]}...')
        article_data = articles.name2articles[entry[1]]
        classification_results = classify_paragraphs_of_articles(article_data, person_classes, classifiers)

        print(f'Writing results to {result_path}...')
        with open(result_path, 'wt') as f:
            json.dump(classification_results, f, indent=2)


def main() -> int:
    classify_articles_into_person_aspects()
    return 0


if __name__ == '__main__':
    main()
