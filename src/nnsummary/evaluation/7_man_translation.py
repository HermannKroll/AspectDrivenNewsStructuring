import csv
import random

from tqdm import tqdm

from nnsummary.config import FILE_EVALUATION_ARTICLE_SNIPPET_TRANSLATION
from nnsummary.news.articles import NewsArticleReader
from nnsummary.translation.translation import Translator

EVAL_7_SAMPLE_SIZE = 100


def eval_article_snippet_translation():
    articles = NewsArticleReader(verbose=False)
    contents_to_sample = []
    print('Collecting all article contents...')
    for p_name, p_articles in tqdm(articles.name2articles.items()):
        for a in p_articles:
            contents_to_sample.extend(a["content"])

    print(f'Collected {len(contents_to_sample)} text snippets')
    print(f'Sampling down to {EVAL_7_SAMPLE_SIZE}...')
    contents_to_sample = random.sample(contents_to_sample, EVAL_7_SAMPLE_SIZE)
    print(f'Sample has ({len(contents_to_sample)}) items')

    translator = Translator()
    print('Translating snippets...')
    contents_translated = []
    for c in tqdm(contents_to_sample):
        c_trans = translator.translate_nl_to_en(c)
        contents_translated.append(c_trans)

    translator.commit_cache()

    print(f'Writing article translations to {FILE_EVALUATION_ARTICLE_SNIPPET_TRANSLATION}...')
    with open(FILE_EVALUATION_ARTICLE_SNIPPET_TRANSLATION, 'wt') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["snippet_nl", "snippet_en", "syntactical (good/moderate/bad)", "factual (correct/incorrect)"])

        for content_nl, content_en in zip(contents_to_sample, contents_translated):
            writer.writerow([content_nl.strip(), content_en.strip(), "", ""])

    print('Finished')


def main() -> int:
    eval_article_snippet_translation()
    return 0


if __name__ == '__main__':
    main()
