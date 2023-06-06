from nnsummary.config import CLUSTERING_SAMPLE_SIZE, PATH_EVALUATION_DIR, FILE_EVALUATION_CLUSTERING_MANUAL
from nnsummary.translation.translation import Translator
from nnsummary.wikipedia.people_processor import WikipediaPeopleProcessor


def eval_clustering_titles():
    wiki = WikipediaPeopleProcessor()
    translator = Translator()
    section2texts_frequent = {k: v for k, v in wiki.section2texts.items() if len(v) >= CLUSTERING_SAMPLE_SIZE}
    section_titles = sorted({t.strip().lower() for t in section2texts_frequent})
    section_titles_en = set()
    for t in section_titles:
        trans = translator.translate_nl_to_en(t).strip().lower()
        if trans:
            section_titles_en.add(f'{t} [[translation: {trans}]]')
        else:
            section_titles_en.add(f'nl_{t}')

    section_titles_en = sorted(section_titles_en)
    print(f'Writing clustering results to {FILE_EVALUATION_CLUSTERING_MANUAL}')
    with open(FILE_EVALUATION_CLUSTERING_MANUAL, 'wt') as f:
        text = '\n'.join([t for t in section_titles_en])
        f.write(text)

    translator.commit_cache()


def main() -> int:
    eval_clustering_titles()
    return 0


if __name__ == '__main__':
    main()
