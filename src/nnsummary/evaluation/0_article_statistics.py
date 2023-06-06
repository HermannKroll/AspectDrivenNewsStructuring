import json

from nnsummary.config import FILE_EVALUATION_ARTICLE_STATISTICS
from nnsummary.news.articles import NewsArticleReader


def eval_article_statistics():
    articles = NewsArticleReader(verbose=True)
    print(f'Writing article statistics to {FILE_EVALUATION_ARTICLE_STATISTICS}...')
    with open(FILE_EVALUATION_ARTICLE_STATISTICS, 'wt') as f:
        json.dump(articles.statistics, f)
    print('Finished')


def main() -> int:
    eval_article_statistics()
    return 0


if __name__ == '__main__':
    main()
