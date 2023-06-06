import json

import torch
import os

from tqdm import tqdm
from transformers import LEDTokenizer, LEDForConditionalGeneration
from nnsummary.config import MULTI_DOCUMENT_SUMMARIZATION_MODEL, RESULT_DIR, MULTI_DOCUMENT_SUMMARY_TOP_K, UTILIZE_GPU, \
    NEW_ARTICLE_PERSON_INFORMATION, LM_CLASSIFIED_ARTICLE_POSTFIX, LM_CLASSIFIER_PREFIX, MULTI_DOCUMENT_PREFIX, \
    MULTI_DOCUMENT_SNIPPET_SIZE, MULTI_DOCUMENT_SUMMARY_MIN_K, NEWS_ARTICLE_SNIPPET_MIN_LEN
from nnsummary.news.articles import NewsArticleReader
from nnsummary.summary.util import generate_snippet_excerpt
from nnsummary.translation.translation import Translator
from nnsummary.wikipedia.clustering import WikipediaSectionClustering


class Summarizer:

    def __init__(self, translator):
        self.model = LEDForConditionalGeneration.from_pretrained(MULTI_DOCUMENT_SUMMARIZATION_MODEL)
        if UTILIZE_GPU:
            self.model.to("cuda")
        self.tokenizer = LEDTokenizer.from_pretrained(MULTI_DOCUMENT_SUMMARIZATION_MODEL)
        print("Model input size: ", self.tokenizer.model_max_length)
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id
        self.DOCSEP_TOKEN_ID = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        self.translator = translator

    def generate_input_ids(self, documents):
        input_ids_all = []
        for data in documents:
            all_docs = data.split("|||||")[:-1]
            for i, doc in enumerate(all_docs):
                doc = doc.replace("\n", " ")
                doc = " ".join(doc.split())
                all_docs[i] = doc

            #### concat with global attention on doc-sep
            input_ids = []
            for doc in all_docs:
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=True,
                        max_length=4096 // len(all_docs),
                    )[1:-1]
                )
                input_ids.append(self.DOCSEP_TOKEN_ID)
            input_ids = (
                    [self.tokenizer.bos_token_id]
                    + input_ids
                    + [self.tokenizer.eos_token_id]
            )
            input_ids_all.append(torch.tensor(input_ids))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_all, batch_first=True, padding_value=self.PAD_TOKEN_ID
        )
        return input_ids

    def summarize_texts(self, texts):
        text_list = [' ||||| '.join([t for t in texts])]
        input_ids = self.generate_input_ids(text_list)
        if UTILIZE_GPU:
            input_ids = input_ids.to("cuda")
        global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == self.DOCSEP_TOKEN_ID] = 1
        input_texts_from_ids = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=1024,
            num_beams=5,
        )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
        return generated_str[0], input_texts_from_ids[0]

    def translate_and_summarize_texts(self, texts):
        translated = list([self.translator.translate_nl_to_en(t) for t in texts])
        # print(translated)
        return self.summarize_texts(translated)


def summarize_classified_news_articles():
    articles = NewsArticleReader()
    translator = Translator()
    summarizer = Summarizer(translator)
    clustering = WikipediaSectionClustering()

    print('--' * 60)
    print(f'Summarizing {len(NEW_ARTICLE_PERSON_INFORMATION)} persons...')
    print('--' * 60)

    for person in NEW_ARTICLE_PERSON_INFORMATION:
        person_name = person[1]
        cls_path = os.path.join(RESULT_DIR, person_name + LM_CLASSIFIED_ARTICLE_POSTFIX)
        if not os.path.isfile(cls_path):
            print(f'Skipping missing file {cls_path}')
            continue

        result_path = os.path.join(RESULT_DIR, f'{MULTI_DOCUMENT_PREFIX}{person_name}.json')
        if os.path.isfile(result_path):
            print(f'Skipping person because summary file has already been computed ({result_path})')
            continue

        s_result = {"name": person_name, "roles": []}
        person_classes = []
        with open(cls_path, 'rt') as f:
            cls_data = json.load(f)
            for person_class_key in tqdm(cls_data):
                person_class = person_class_key.replace(f'{LM_CLASSIFIER_PREFIX}_', '')
                person_classes.append(person_class)
                class_aspects, class_aspects_with_clustering = [], []
                sections_for_class = []
                for aspect in cls_data[person_class_key]:
                    if aspect == "no class":
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

                        if len(c) < NEWS_ARTICLE_SNIPPET_MIN_LEN:
                            raise ValueError(f'Classified snippet {c} is shorter than {NEWS_ARTICLE_SNIPPET_MIN_LEN}')

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

                    if len(contents_to_summarize) > MULTI_DOCUMENT_SUMMARY_TOP_K:
                        contents_to_summarize = contents_to_summarize[:MULTI_DOCUMENT_SUMMARY_TOP_K]

                    # We know will summarize the content
                    class_aspects.append(aspect)
                    aspect_rewritten = clustering.find_also_name_as_titles(aspect)
                    class_aspects_with_clustering.append(aspect_rewritten)

                    # create the summary
                    texts = list([t[1] for t in contents_to_summarize])

                    summary, summary_input = summarizer.translate_and_summarize_texts(texts)
                    summary_nl = translator.translate_en_to_nl(summary)
                    
                    section_data = {"title": aspect_rewritten, "title_org": aspect, "text": summary,
                                    "text_nl": summary_nl, "articles": []}
                    for art_id, art_snipp, _ in contents_to_summarize:
                        local_art_data = articles.prepare_article_json(person_name, art_id)
                        local_art_data["text"] = generate_snippet_excerpt(art_snipp, person)
                        section_data["articles"].append(local_art_data)

                    # Put it into final result
                    sections_for_class.append(section_data)

                if len(class_aspects) > 0 and len(sections_for_class) > 0:
                    # This person class is now a role
                    s_result["roles"].append(dict(name=person_class,
                                                  sections=class_aspects_with_clustering,
                                                  sections_org=class_aspects,
                                                  section_data=sections_for_class))

        with open(result_path, "wt") as f:
            json.dump(s_result, f)

        # save translation cache
        translator.commit_cache()


def main() -> int:
    summarize_classified_news_articles()
    return 0


if __name__ == '__main__':
    main()
