import csv
import json
import os
import random

from tqdm import tqdm

from nnsummary.config import RESULT_DIR, PATH_EVALUATION_DIR, LM_CLASSIFIED_ARTICLE_POSTFIX, \
    NEW_ARTICLE_PERSON_INFORMATION, LM_CLASSIFIER_PREFIX, EVAL_TEST_SUMMARY_K, MULTI_DOCUMENT_SUMMARY_MIN_K, \
    MULTI_DOCUMENT_SUMMARY_TOP_K, EVAL_SUMMARIES_EXPORT_K, PATH_EVALUATION_SUMMARIES_MANUAL, \
    FILE_EVALUATION_SUMMARIES_MANUAL
from nnsummary.summary.summarizer import Summarizer
from nnsummary.translation.translation import Translator


def eval_export_summaries_for_manual_evaluation():
    print('--' * 60)
    print(f'Selecting data for {len(NEW_ARTICLE_PERSON_INFORMATION)} persons...')
    print('--' * 60)

    candidates_to_summarize = []
    for person in NEW_ARTICLE_PERSON_INFORMATION:
        person_name = person[1]
        cls_path = os.path.join(RESULT_DIR, person_name + LM_CLASSIFIED_ARTICLE_POSTFIX)
        if not os.path.isfile(cls_path):
            print(f'Skipping missing file {cls_path}')
            continue

        with open(cls_path, 'rt') as f:
            cls_data = json.load(f)
            for person_class_key in tqdm(cls_data):
                person_class = person_class_key.replace(f'{LM_CLASSIFIER_PREFIX}_', '')
                for aspect in cls_data[person_class_key]:
                    if "no class" in aspect:
                        continue

                    contents_to_summarize = []
                    for entry_data in cls_data[person_class_key][aspect]:
                        art_id = entry_data["article_id"]
                        content = entry_data["content"]
                        probability = entry_data["probability"]

                        contents_to_summarize.append((art_id, content, probability))

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

                    contents_to_summarize = contents_to_summarize_filtered
                    # we don't have content so skip the summarizing part
                    if len(contents_to_summarize) == 0:
                        continue

                    # Do not summarize if too less entries are available
                    if len(contents_to_summarize) < MULTI_DOCUMENT_SUMMARY_MIN_K:
                        continue

                    # It should be at least 20 docs
                    if len(contents_to_summarize) < MULTI_DOCUMENT_SUMMARY_TOP_K:
                        continue

                    texts = list([t[1] for t in contents_to_summarize])
                    texts = texts[:MULTI_DOCUMENT_SUMMARY_TOP_K]
                    candidates_to_summarize.append((person_name, person_class, aspect, texts))

    print(f'Received {len(candidates_to_summarize)} candidates to summarize')
    print(f'Random sample to {EVAL_SUMMARIES_EXPORT_K}...')
    candidates_to_summarize = random.sample(candidates_to_summarize, EVAL_SUMMARIES_EXPORT_K)

    translator = Translator()
    summarizer = Summarizer(translator)
    print('Summarizing persons...')
    index = 0
    summary_tsv_results = []
    for person_name, person_class, aspect, texts in tqdm(candidates_to_summarize):
        translated_texts = list([translator.translate_nl_to_en(t) for t in texts])
        summary, _ = summarizer.translate_and_summarize_texts(texts)
        summary_nl = translator.translate_en_to_nl(summary)

        summary_tsv_results.append((index, summary_nl, summary, "", "", ""))
        path_for_text_snippets = os.path.join(PATH_EVALUATION_SUMMARIES_MANUAL, f'{index}_text_snippets.txt')
        with open(path_for_text_snippets, 'wt') as f:
            for t, t_trans in zip(texts, translated_texts):
                f.write(t)
                f.write('\n')
                f.write('--' * 60)
                f.write('\n')
                f.write(t_trans)

                f.write('\n\n\n\n')
                f.write('==' * 60)
                f.write('\n')
                f.write('==' * 60)
                f.write('\n')

        index += 1

    # save translation cache
    translator.commit_cache()

    print(f'Writing summaries to {FILE_EVALUATION_SUMMARIES_MANUAL}...')
    with open(FILE_EVALUATION_SUMMARIES_MANUAL, 'wt') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["index", "summary_nl", "summary_en",
                         "score (good/moderate/bad)", "factual (correct/partial/wrong)", "remarks"])

        for s in summary_tsv_results:
            writer.writerow(s)


def main() -> int:
    eval_export_summaries_for_manual_evaluation()
    return 0


if __name__ == '__main__':
    main()


def main() -> int:
    return 0


if __name__ == '__main__':
    main()
