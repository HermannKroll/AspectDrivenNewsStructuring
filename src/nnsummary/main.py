from nnsummary.classification.apply_models import classify_articles_into_person_aspects
from nnsummary.summary.summarizer import summarize_classified_news_articles

DO_ARTICLE_CLASSIFICATION = True
DO_ARTICLE_SUMMARIZATION = True


def main() -> int:
    if DO_ARTICLE_CLASSIFICATION:
        classify_articles_into_person_aspects()
    if DO_ARTICLE_SUMMARIZATION:
        summarize_classified_news_articles()

    return 0


if __name__ == '__main__':
    main()
