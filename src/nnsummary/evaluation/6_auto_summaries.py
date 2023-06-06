import json
import os

from tqdm import tqdm

from nnsummary.config import RESULT_DIR, PATH_EVALUATION_DIR, LM_CLASSIFIED_ARTICLE_POSTFIX, \
    NEW_ARTICLE_PERSON_INFORMATION, LM_CLASSIFIER_PREFIX, EVAL_TEST_SUMMARY_K, MULTI_DOCUMENT_SUMMARY_MIN_K
from nnsummary.summary.summarizer import Summarizer
from nnsummary.translation.translation import Translator


def eval_export_summaries_of_different_length():
    translator = Translator()
    summarizer = Summarizer(translator)

    print('--' * 60)
    print(f'Summarizing {len(NEW_ARTICLE_PERSON_INFORMATION)} persons...')
    print('--' * 60)

    for person in NEW_ARTICLE_PERSON_INFORMATION:
        person_name = person[1]
        cls_path = os.path.join(RESULT_DIR, person_name + LM_CLASSIFIED_ARTICLE_POSTFIX)
        if not os.path.isfile(cls_path):
            print(f'Skipping missing file {cls_path}')
            continue

        result_path = os.path.join(PATH_EVALUATION_DIR, f'5_eval_summaries_{person_name}.json')
        if os.path.isfile(result_path):
            print(f'Skipping already computed file {result_path}')
            continue

        s_result = {"name": person_name, "roles": []}
        with open(cls_path, 'rt') as f:
            cls_data = json.load(f)
            for person_class_key in tqdm(cls_data):
                person_class = person_class_key.replace(f'{LM_CLASSIFIER_PREFIX}_', '')
                sections_for_class = []
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

                    # generate other summaries:
                    section_data = {"summaries_k": {}, "title": aspect}
                    texts = list([t[1] for t in contents_to_summarize])
                    for k in EVAL_TEST_SUMMARY_K:
                        texts_k = texts
                        if len(texts_k) > k:
                            texts_k = texts_k[:k]

                        translated_texts_k = list([translator.translate_nl_to_en(t) for t in texts_k])
                        summary_k, summary_input = summarizer.translate_and_summarize_texts(texts_k)
                        summary_input_nl = translator.translate_en_to_nl(summary_input)
                        summary_k_nl = translator.translate_en_to_nl(summary_k)

                        section_data["summaries_k"][k] = {"summary": summary_k, "summary_nl": summary_k_nl,
                                                          "summary_input_text": summary_input,
                                                          "summary_input_text_nl:": summary_input_nl,
                                                          "org_texts": translated_texts_k, "org_texts_nl": texts_k}

                    # Put it into final result
                    sections_for_class.append(section_data)

                if len(sections_for_class) > 0:
                    # This person class is now a role
                    s_result["roles"].append(dict(name=person_class,
                                                  summary=sections_for_class))

        with open(result_path, "wt") as f:
            json.dump(s_result, f)

        # save translation cache
        translator.commit_cache()


def main() -> int:
    eval_export_summaries_of_different_length()
    return 0


if __name__ == '__main__':
    main()
