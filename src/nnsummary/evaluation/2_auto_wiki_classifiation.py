import json
import os

import numpy
from tqdm import tqdm

from nnsummary.classification.language_model import LanguageModelTrainer
from nnsummary.config import PATH_LM_MODEL_TRAIN_STATISTICS, LM_CLASSIFIER_PREFIX, \
    FILE_EVALUATION_ARTICLE_WIKI_CLASSIFICATION_QUALITY
from nnsummary.wikipedia.aspect_mining import AspectMining


def generate_quality_report(setting, entries):
    avg_p = round(float(numpy.mean([e[1] for e in entries])), 2)
    avg_p_std = round(float(numpy.std([e[1] for e in entries])), 2)

    avg_r = round(float(numpy.mean([e[2] for e in entries])), 2)
    avg_r_std = round(float(numpy.std([e[2] for e in entries])), 2)

    avg_f1 = round(float(numpy.mean([e[3] for e in entries])), 2)
    avg_f1_std = round(float(numpy.std([e[3] for e in entries])), 2)

    avg_acc = round(float(numpy.mean([e[4] for e in entries])), 2)
    avg_acc_std = round(float(numpy.std([e[4] for e in entries])), 2)

    avg_aspects = round(float(numpy.mean([e[5] for e in entries])), 2)
    avg_aspects_std = round(float(numpy.std([e[5] for e in entries])), 2)

    avg_samples = round(float(numpy.mean([e[6] for e in entries])))
    avg_samples_std = round(float(numpy.std([e[6] for e in entries])))

    print(f'{setting} & ${avg_aspects}\pm{avg_aspects_std}$ & '
          f' ${avg_samples}\pm{avg_samples_std}$ &'
          f' ${avg_p}\pm{avg_p_std}$ &'
          f' ${avg_r}\pm{avg_r_std}$ &'
          f' ${avg_f1}\pm{avg_f1_std}$ &'
          f' ${avg_acc}\pm{avg_acc_std}$ \\\\')

    return dict(avg_precision=avg_p, avg_precision_std=avg_p_std,
                avg_recall=avg_r, average_recall_std=avg_r_std,
                avg_f1=avg_f1, avg_f1_std=avg_f1_std,
                avg_accuracy=avg_acc, avg_accuracy_std=avg_acc_std,
                avg_aspects=avg_aspects, avg_aspects_std=avg_aspects_std,
                avg_samples=avg_samples, avg_samples_std=avg_samples_std)


def eval_wiki_article_classification_quality():
    mining = AspectMining()
    print(f'Discovering {PATH_LM_MODEL_TRAIN_STATISTICS}...')
    model_training_reports = []
    for filename in os.listdir(PATH_LM_MODEL_TRAIN_STATISTICS):
        if filename.lower().endswith(".json"):
            model_training_reports.append(os.path.join(PATH_LM_MODEL_TRAIN_STATISTICS, filename))
    print(f'Found {len(model_training_reports)} training reports')

    model_trainer = LanguageModelTrainer()
    model_stats = []
    for entry in tqdm(model_training_reports):
        with open(entry, 'rt') as f:
            data = json.load(f)

            # get the name of the classifier
            model_class = entry.split('/')[-1].replace(f'results_{LM_CLASSIFIER_PREFIX}_', '').replace('.json',
                                                                                                       '').strip()

            aspects = sorted(mining.get_person_aspects_for_class(model_class))
            no_aspects = len(aspects)
            # Works because of fixed random seeds
            data_set = model_trainer.random_sample_training_data(aspects, model_class, verbose=False)
            no_samples = data_set.get_train_size()

            m_precision = data["test"]["macro_precision"]
            m_recall = data["test"]["macro_recall"]
            m_f1 = data["test"]["macro_f1"]
            accuracy = data["test"]["eval_accuracy"]

            model_stats.append((model_class, m_precision, m_recall, m_f1, accuracy, no_aspects, no_samples))

    print('Sorting for macro precision...')
    model_stats.sort(key=lambda x: x[1], reverse=True)

    nice_entries = []
    for e in model_stats:
        nice_entries.append({"person_class": e[0],
                             "macro_precision": e[1],
                             "macro_recall": e[2],
                             "macro_f1": e[3],
                             "accuracy": e[4],
                             "no_aspects": e[5],
                             "no_samples": e[6]})
    results = {"data": nice_entries, "settings": {}}

    settings = [("Top-5", model_stats[:5]), ("Top-10", model_stats[:10]),
                ("Worst-5", model_stats[-5:]), ("Worst-10", model_stats[-10:]),
                ("All", model_stats)]
    for s, entries in settings:
        results["settings"][s] = generate_quality_report(s, entries)

    print(f'Write statistics to {FILE_EVALUATION_ARTICLE_WIKI_CLASSIFICATION_QUALITY}')
    with open(FILE_EVALUATION_ARTICLE_WIKI_CLASSIFICATION_QUALITY, 'wt') as f:
        json.dump(results, f)


def main() -> int:
    eval_wiki_article_classification_quality()
    return 0


if __name__ == '__main__':
    main()
