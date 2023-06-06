import json
import os

import numpy
from tqdm import tqdm

from nnsummary.classification.language_model import LanguageModelTrainer
from nnsummary.config import NEW_ARTICLE_PERSON_INFORMATION, RESULT_DIR, LM_CLASSIFIED_ARTICLE_POSTFIX, \
    LM_CLASSIFIER_PREFIX, CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH, FILE_EVALUATION_ARTICLE_CLASSIFICATION_STATISTICS, \
    EVAL_TEST_SUMMARY_K, PATH_LM_MODEL_TRAIN_STATISTICS, FILE_EVALUATION_ARTICLE_WIKI_CLASSIFICATION_QUALITY
from nnsummary.wikipedia.aspect_mining import AspectMining
from nnsummary.wikipedia.category_tree import CategoryTree
from nnsummary.wikipedia.people_processor import WikipediaPeopleProcessor


def eval_count_article_classification():
    wiki = WikipediaPeopleProcessor()
    cat_tree = CategoryTree()
    print('--' * 60)
    print(f'Analyzing {len(NEW_ARTICLE_PERSON_INFORMATION)} persons...')
    print('--' * 60)

    results = {}
    more_than_x_samples = {k: 0 for k in EVAL_TEST_SUMMARY_K}
    for person in NEW_ARTICLE_PERSON_INFORMATION:
        person_name = person[1]
        cls_path = os.path.join(RESULT_DIR, person_name + LM_CLASSIFIED_ARTICLE_POSTFIX)
        if not os.path.isfile(cls_path):
            print(f'Skipping missing file {cls_path}')
            continue

        if person_name not in results:
            results[person_name] = {}

        if person[4] not in wiki.person2categories:
            print(f'Person does not have categories assigned -> use {cat_tree.root}')
            person_classes = {cat_tree.root}
        else:
            person_classes = cat_tree.filter_occupations_by_level(wiki.person2categories[person[4]],
                                                                  CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH)

        no_person_classes = len(person_classes)
        results[person_name]["no_person_classes"] = no_person_classes
        results[person_name]["person_classes"] = sorted(list(person_classes))

        classified_snippets = 0
        more_than_x_samples_per_person = {k: 0 for k in EVAL_TEST_SUMMARY_K}
        no_aspects, no_aspects_min = 0, 0
        classified_snippets_all_aspect_list = []
        with open(cls_path, 'rt') as f:
            cls_data = json.load(f)
            for person_class_key in tqdm(cls_data):
                person_class = person_class_key.replace(f'{LM_CLASSIFIER_PREFIX}_', '')
                classified_snippets_per_class = 0

                if person_class not in results[person_name]:
                    results[person_name][person_class] = {}

                for aspect in cls_data[person_class_key]:
                    if aspect == "no class":
                        continue

                    no_aspects += 1

                    contents_to_summarize = []
                    for entry_data in cls_data[person_class_key][aspect]:
                        art_id = entry_data["article_id"]
                        content = entry_data["content"]
                        probability = entry_data["probability"]

                        contents_to_summarize.append((art_id, content, probability))

                    # we don't have content so skip the summarizing part
                    if len(contents_to_summarize) == 0:
                        continue

                    # compute max probability per doc
                    contents_to_summarize.sort(key=lambda x: x[2], reverse=True)
                    # Only take the best snippet per article
                    contents_to_summarize_filtered = []
                    visited_aids = set()
                    for aid, c, p in contents_to_summarize:
                        if not c.strip():
                            continue

                        if aid in visited_aids:
                            continue
                        visited_aids.add(aid)
                        contents_to_summarize_filtered.append((aid, c, p))

                    # Count how many are above ..
                    for k in more_than_x_samples:
                        if len(contents_to_summarize_filtered) >= k:
                            more_than_x_samples[k] += 1
                            more_than_x_samples_per_person[k] += 1

                    no_aspects_min += 1
                    results[person_name][person_class][aspect] = {}
                    # No of articles per
                    no_classified_snippets = len(contents_to_summarize)

                    classified_snippets_all_aspect_list.append(no_classified_snippets)
                    classified_snippets_per_class += no_classified_snippets
                    classified_snippets += no_classified_snippets
                    results[person_name][person_class][aspect]["no_classified_snippets"] = no_classified_snippets

            results[person_name][person_class]["no_classified_snippets_in_sum"] = classified_snippets_per_class

        results[person_name]["summary_statistics"] = more_than_x_samples_per_person
        results[person_name]["no_classified_snippets"] = classified_snippets
        results[person_name]["no_aspects"] = no_aspects
        results[person_name]["no_aspects_min_>0"] = no_aspects_min
        if no_aspects_min > 0:
            results[person_name]["avg_no_classified_snippets_per_aspect_min_>0"] = round(
                float(classified_snippets / no_aspects_min))
            results[person_name]["avg_no_classified_snippets_per_aspect_min_>0_std"] = round(float(numpy.std(
                classified_snippets_all_aspect_list)))
        else:
            results[person_name]["avg_no_classified_snippets_per_aspect_min_>0"] = 0
            results[person_name]["avg_no_classified_snippets_per_aspect_min_>0_std"] = 0

        results[person_name]["no_aspects_list"] = sorted(classified_snippets_all_aspect_list, reverse=True)
        results[person_name]["no_aspects_min"] = min(classified_snippets_all_aspect_list)
        results[person_name]["no_aspects_max"] = max(classified_snippets_all_aspect_list)

    print('--' * 60)
    print('--' * 60)
    print('Latex Code')
    print('--' * 60)
    print('--' * 60)
    for person_name in results:
        print(f'{person_name}')
        no_aspects = results[person_name]["no_aspects_min_>0"]
        no_classified_snippets = results[person_name]["no_classified_snippets"]
        no_snip_per_aspect_mean = results[person_name]["avg_no_classified_snippets_per_aspect_min_>0"]
        no_snip_per_aspect_mean_std = results[person_name]["avg_no_classified_snippets_per_aspect_min_>0_std"]
        no_snip_per_aspect_min = results[person_name]["no_aspects_min"]
        no_snip_per_aspect_max = results[person_name]["no_aspects_max"]
        print(f'  & {no_classified_snippets} & {no_aspects} & '
              f'${no_snip_per_aspect_mean}\pm{no_snip_per_aspect_mean_std}$ & '
              f'{no_snip_per_aspect_min} & {no_snip_per_aspect_max} \\\\')

    results["summary_statistics"] = more_than_x_samples
    print(f'Writing results to {FILE_EVALUATION_ARTICLE_CLASSIFICATION_STATISTICS}')
    with open(FILE_EVALUATION_ARTICLE_CLASSIFICATION_STATISTICS, 'wt') as f:
        json.dump(results, f)
    print('Finished')


def main() -> int:
    eval_count_article_classification()
    return 0


if __name__ == '__main__':
    main()
