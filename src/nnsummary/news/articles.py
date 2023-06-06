import ast
import os
import json
import nltk
from tqdm import tqdm

from nnsummary.config import NEWS_ARTICLE_PATH, NEWS_ARTICLE_PERSON_NAME_FILTER, \
    NEWS_ARTICLE_SENTENCE_SLIDING_WINDOW_SIZE, NEW_ARTICLE_PERSON_INFORMATION, NEWS_ARTICLE_SNIPPET_MIN_LEN


class NewsArticleReader:

    def __init__(self, verbose=False):
        self.name2articles = {}
        self.name2artid2article = {}
        self.statistics = {}
        for entry in NEW_ARTICLE_PERSON_INFORMATION:
            article_data = self.load_articles_for_person(entry, verbose=verbose)
            if verbose:
                print(f'{len(article_data)} {entry[1]} news articles load')

            person_name = entry[1]
            self.name2articles[person_name] = article_data
            self.name2artid2article[person_name] = {}
            for art in article_data:
                self.name2artid2article[person_name][art["record_identifier"]] = art

    def load_articles_for_person(self, person_entry, verbose=True):
        path = os.path.join(NEWS_ARTICLE_PATH, person_entry[0])
        if verbose:
            print(f'Loading articles from {path}')
        data = []
        person_name = person_entry[1]
        self.statistics[person_name] = {}

        person_name_parts = list([f'{n.lower().strip()}' for n in person_entry[5]])
        if verbose:
            print('Person name parts: ', person_name_parts)
        birth, death = person_entry[2], person_entry[3]
        count, count_loaded, skipped, no_contents = 0, 0, 0, 0
        skipped_missing_metadata, skipped_len, skipped_name, skipped_date = 0, 0, 0, 0
        snippets_too_short = 0
        with open(path, 'rt') as f:
            json_data = json.load(f)
            for art_data in tqdm(json_data):
                count += 1
                if "title" not in art_data or "content" not in art_data:
                    skipped += 1
                    skipped_missing_metadata += 1
                    continue
                if len(art_data["content"]) < NEWS_ARTICLE_SNIPPET_MIN_LEN:
                    skipped += 1
                    skipped_len += 1
                    continue
                # just consider the year of the article data
                date = int(art_data["date"].split('-')[0])
                # only news articles that are in the range between birth and death
                if date < birth or date > death:
                    skipped += 1
                    skipped_date += 1
                    continue

                count_loaded += 1

                # interpret content field as a list of strings
                art_data["content"] = ast.literal_eval(art_data["content"])
                # art_data["content"] = list([c.strip() for c in art_data["content"]
                #                            if len(c.strip()) >= NEWS_ARTICLE_PARAGRAPH_MIN_LEN])

                # Interpret everything as one long paragraph
                art_data["content"] = ' '.join([c for c in art_data["content"]])
                # Split the content into sentences - only use snippets
                # that contain the person name
                #  print(f'Text has {len(art_data["content"] )} characters')
                if NEWS_ARTICLE_PERSON_NAME_FILTER:
                    relevant_snipptes = []
                    content = art_data["content"]
                    # for content in art_data["content"]:
                    sentences = list(nltk.sent_tokenize(content))
                    for idx, s in enumerate(sentences):
                        for pn_part in person_name_parts:
                            s_to_check = f'{s}'.lower()
                            if pn_part in s_to_check:
                                # match
                                relevant_sentences = []
                                w = NEWS_ARTICLE_SENTENCE_SLIDING_WINDOW_SIZE
                                # because of exclusive
                                for i in range(-w, w + 1):
                                    if 0 <= idx + i < len(sentences):
                                        relevant_sentences.append(sentences[idx + i])

                                rel_snip = ' '.join([rs for rs in relevant_sentences])
                                if len(rel_snip) >= NEWS_ARTICLE_SNIPPET_MIN_LEN:
                                    relevant_snipptes.append(rel_snip)
                                    # One part of the sentences is enough
                                    break
                                else:
                                    snippets_too_short += 1

                    # Set relevant snippts as the new content
                    # Those snippets contain a part of the person name
                    art_data["content"] = relevant_snipptes

                if len(art_data["content"]) == 0:
                    if NEWS_ARTICLE_PERSON_NAME_FILTER:
                        skipped_name += 1
                    skipped += 1
                    continue
                no_contents += len(art_data["content"])
                data.append(art_data)

        # Set statistics
        self.statistics[person_name]["skipped"] = skipped
        self.statistics[person_name]["count"] = count
        self.statistics[person_name]["count_loaded"] = count_loaded
        self.statistics[person_name]["snippets"] = no_contents
        self.statistics[person_name]["used_name_parts"] = person_name_parts
        self.statistics[person_name]["skipped_bc_len"] = skipped_len
        self.statistics[person_name]["skipped_bc_name"] = skipped_name
        self.statistics[person_name]["skipped_bc_year"] = skipped_date
        self.statistics[person_name]["skipped_bc_metadata"] = skipped_missing_metadata
        self.statistics[person_name]["snippets_to_short"] = snippets_too_short
        self.statistics[person_name]["snippets_min_length"] = NEWS_ARTICLE_SNIPPET_MIN_LEN

        print(f'{skipped} of {count} articles have been skipped in ({count_loaded} articles loaded from {path})')
        print(f'\t=> {no_contents} paragraphs loaded')
        if verbose:
            print(f'\tDetails: {skipped_len} because of minimum character length')
            print(f'\tDetails: {skipped_missing_metadata} because of missing metadata')
            print(f'\tDetails: {skipped_date} because date is not between birth and death')
            if NEWS_ARTICLE_PERSON_NAME_FILTER:
                print(f'\tDetails: {skipped_name} because of name was not found in content')
                print(f'\tDetails: {snippets_too_short} snippets were too short (< {NEWS_ARTICLE_SNIPPET_MIN_LEN})')
        return data

    def get_article(self, person_name, record_identifier):
        return self.name2artid2article[person_name][record_identifier]

    def prepare_article_json(self, person_name, record_identifier):
        art = self.get_article(person_name, record_identifier)
        return {"newspaper": art["newspaper"] + f' ({art["date"]})', "headline": art["title"],
                "url": art["article_identifier"], "text": "This could be a snippet."}


def main() -> int:
    NewsArticleReader(verbose=True)
    return 0


if __name__ == '__main__':
    main()
