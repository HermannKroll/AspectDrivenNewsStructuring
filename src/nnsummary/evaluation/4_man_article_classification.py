import csv
import json
import os
import random

from tqdm import tqdm

from nnsummary.config import FILE_EVALUATION_ARTICLE_CLASSIFICATION_MANUAL, LM_CLASSIFIED_ARTICLE_POSTFIX, \
    NEW_ARTICLE_PERSON_INFORMATION, RESULT_DIR, LM_CLASSIFIER_PREFIX, \
    FILE_EVALUATION_ARTICLE_CLASSIFICATION_MANUAL_BALANCED
from nnsummary.translation.translation import Translator
from nnsummary.wikipedia.clustering import WikipediaSectionClustering

example_row = ["person", "background", "snippet in nl", "snippet in en", "yes", ""]

TOP_K_SAMPLE_PER_CLASS_AND_ASPECT = 5
SAMPLE_SIZE = 100


def eval_sample_article_classification(balanced=False):
    translator = Translator()
    clustering = WikipediaSectionClustering()
    person_names = [p[1] + LM_CLASSIFIED_ARTICLE_POSTFIX for p in NEW_ARTICLE_PERSON_INFORMATION]
    cls_res_paths = list([os.path.join(RESULT_DIR, c) for c in person_names])

    print('--' * 60)
    print(f'Collecting snippets from {person_names}...')
    print('--' * 60)

    classfied_contents = []
    for person_name, cls_path in zip(person_names, cls_res_paths):
        if not os.path.isfile(cls_path):
            print(f'Skipping missing file {cls_path}')
            continue

        person_name = person_name.replace('_classified.json', '')
        person_classes = []
        with open(cls_path, 'rt') as f:
            print(f'Reading {cls_path}...')
            cls_data = json.load(f)
            for person_class_key in tqdm(cls_data):
                person_class = person_class_key.replace(f'{LM_CLASSIFIER_PREFIX}_', '')
                person_classes.append(person_class)
                for aspect in cls_data[person_class_key]:
                    for entry_data in cls_data[person_class_key][aspect]:
                        art_id = entry_data["article_id"]
                        content = entry_data["content"]
                        probability = entry_data["probability"]
                        aspect_rewritten = clustering.find_also_name_as_titles(aspect)

                        classfied_contents.append((person_name, probability, person_class, aspect_rewritten, art_id, content))

    # sort by probability
    if balanced:
        print(f'Could export {len(classfied_contents)} examples...')
        print('Sorting...')
        classfied_contents.sort(key=lambda x: x[1], reverse=True)
        person_class_aspect2content = {}
        print('Aggregating by person class and aspect...')
        print(f'Taking {TOP_K_SAMPLE_PER_CLASS_AND_ASPECT} per person class and aspect... ')
        entries_to_sample = list()
        for entry in classfied_contents:
            # person class + aspect
            key = (entry[2], entry[3])
            if key not in person_class_aspect2content:
                person_class_aspect2content[key] = list()

            # Only take 5 per aspect and class
            if len(person_class_aspect2content[key]) > TOP_K_SAMPLE_PER_CLASS_AND_ASPECT:
                continue
            # Ok less than 5
            # translate content

            person_class_aspect2content[key].append(entry)
            entries_to_sample.append(entry)

        print(f'Found {len(entries_to_sample)} candidates to sample')
        print(f'Sampling down to {SAMPLE_SIZE}...')
        selected_sample = random.sample(entries_to_sample, SAMPLE_SIZE)
    else:
        print(f'Found {len(classfied_contents)} candidates to sample')
        print(f'Sampling down to {SAMPLE_SIZE}...')
        selected_sample = random.sample(classfied_contents, SAMPLE_SIZE)
    results = []
    index = 1
    for person_name, p, pc, a, aid, content in selected_sample:
        content_en = translator.translate_nl_to_en(content).strip()
        pc_en = translator.translate_nl_to_en(pc).strip()
        # translate part by part
        a_en = ' | '.join([translator.translate_nl_to_en(a_part).strip() for a_part in a.split(' | ')])
        results.append((index, p, person_name, f'{pc} [{pc_en}]', f'{a} [{a_en}]', content, content_en))
        index += 1

    translator.commit_cache()
    return results


def eval_manual_article_classification():
    samples = eval_sample_article_classification(balanced=False)

    print(f'Writing article classifications to {FILE_EVALUATION_ARTICLE_CLASSIFICATION_MANUAL}...')
    with open(FILE_EVALUATION_ARTICLE_CLASSIFICATION_MANUAL, 'wt') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["index", "probability", "person name", "person class", "labels", "snippet_nl", "snippet_en", "yes/no", "why (if no)"])

        for s in samples:
            writer.writerow(s)

    print('Finished')

    samples_balanced = eval_sample_article_classification(balanced=True)
    print(f'Writing article classifications to {FILE_EVALUATION_ARTICLE_CLASSIFICATION_MANUAL_BALANCED}...')
    with open(FILE_EVALUATION_ARTICLE_CLASSIFICATION_MANUAL_BALANCED, 'wt') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(
            ["index", "probability", "person name", "person class", "labels", "snippet_nl", "snippet_en", "yes/no",
             "why (if no)"])

        for s in samples_balanced:
            writer.writerow(s)

    print('Finished')

def main() -> int:
    eval_manual_article_classification()
    return 0


if __name__ == '__main__':
    main()
