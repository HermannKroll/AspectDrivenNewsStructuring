import os

DATA_DIR = "/home/kroll/TPDL2023/"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULT_DIR = os.path.join(DATA_DIR, "results")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
RESOURCE_DIR = os.path.join(ROOT_DIR, "resources")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(RESOURCE_DIR):
    os.makedirs(RESOURCE_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

FILE_WIKIPEDIA_CATEGORIES = os.path.join(RESOURCE_DIR, "categories.json")

PATH_WIKIPEDIA_JSONS = os.path.join(DATA_DIR, "people")

FILE_CACHE_WIKIPEDIA_PEOPLE_DATA = os.path.join(CACHE_DIR, "people_data.pkl")

FILE_CACHE_WIKIPEDIA_SECTION_CLUSTERING = os.path.join(CACHE_DIR, "wikipedia_section_clustering.pkl")
CLUSTERING_SAMPLE_SIZE = 100
CLUSTERING_SECTION_SIMILARITY = 0.95

WIKIPEDIA_MIN_SUMMARY_LENGTH = 150
WIKIPEDIA_IN_SECTIONS = 2
SECTION_MIN_LENGTH = 100
SECTIONS_TO_IGNORE = {'externe links', 'externe link', 'zie ook', 'referenties',
                      'bronnen', 'literatuur en bronnen', 'literatuur'}

FILE_CACHE_ASPECT_MINING = os.path.join(CACHE_DIR, "aspect_mining.pkl")
ASPECT_MINING_MINIMUM_TEXTS: int = 100
ASPECT_MINING_MINIMUM_SUPPORT = 0.05

CATEGORY_PERSON_CLASS_ONTOLOGICAL_DEPTH = 2

NEWS_ARTICLE_PATH = os.path.join(DATA_DIR, "articles")
NEWS_ARTICLE_SNIPPET_MIN_LEN = 50
NEWS_ARTICLE_PERSON_NAME_FILTER = True
NEWS_ARTICLE_SENTENCE_SLIDING_WINDOW_SIZE = 1  # One sentence before and one after

# ARTICLE JSON FILE, full name, birth, death, wikipedia name, list of synonyms for snippet generation
NEW_ARTICLE_PERSON_INFORMATION = [
    ("WinstonChurchill.json", 'Winston Churchill', 1874, 1965, "winston churchill", ["winston", "churchill"]),
    ('LeopoldIII.json', "Leopold III", 1901, 1983, "leopold iii van belgië", ["leopold iii", "prins leopold"]),
    ("Wilhelmina.json", 'Wilhelmina van Oranje-Nassau', 1880, 1962, "wilhelmina der nederlanden", ["wilhelmina"]),
    ('HannieSchaft.json', 'Hannie Schaft', 1920, 1945, "hannie schaft", ["hannie", "schaft", "jannetje"]),
    ('DwightEisenhower.json', 'Dwight Eisenhower', 1890, 1969, "dwight d. eisenhower", ["dwight", "eisenhower"]),
    ("AnneFrank.json", 'Anne Frank', 1929, 1945, "anne frank", ["anne", "frank"]),
    ('FransGoedhart.json', 'Frans Goedhart', 1904, 1990, "frans goedhart", ["frans", "goedhart", "pieter", "hoen"]),
    ('SimonVestdijk.json', "Simon Vestdijk", 1898, 1971, "simon vestdijk", ["simon", "vestdijk"]),
    ('FranklinRoosevelt.json', "Franklin Roosevelt", 1882, 1945, "franklin delano roosevelt",
     ["franklin", "delano", "roosevelt"])
]

MIN_ASPECTS_TO_TRAIN = 3

PATH_LM_MODEL_TRAIN_STATISTICS = os.path.join(DATA_DIR, "model_training")
if not os.path.exists(PATH_LM_MODEL_TRAIN_STATISTICS):
    os.makedirs(PATH_LM_MODEL_TRAIN_STATISTICS)

PATH_LM_MODEL_OUTPUT_DIR = os.path.join(DATA_DIR, "models")
LM_LEARNING_RATES = [1e-3, 1e-4, 1e-5]
LM_NUM_TRAIN_EPOCHS = [5]  # 3
LM_WEIGHT_DECAYS = [0.1, 0.2]

LM_TRAIN_RATIO = 0.8  # means 0.2 test
LM_VAL_TEST_RATIO = 0.5

LM_RANDOM_SEED = 12345

LM_TO_TRAIN = ["DTAI-KULeuven/robbert-2022-dutch-base"]
LM_CLASSIFIER_PREFIX = "DTAI-KULeuven-robbert-2022-dutch-base"
LM_CLASSIFIED_ARTICLE_POSTFIX = "_classified.json"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
UTILIZE_GPU = True

PATH_TRANSLATION_NL2EN = os.path.join(DATA_DIR, "opus_nlen")
PATH_TRANSLATION_EN2NL = os.path.join(DATA_DIR, "opus_ennl")

# Alternative "allenai/PRIMERA"
MULTI_DOCUMENT_SUMMARIZATION_MODEL = "allenai/PRIMERA-multinews"

MULTI_DOCUMENT_SUMMARY_MIN_K = 5
MULTI_DOCUMENT_SUMMARY_TOP_K = 20
MULTI_DOCUMENT_PREFIX = "summarized_"
MULTI_DOCUMENT_SNIPPET_SIZE = 160

PATH_EVALUATION_DIR = os.path.join(RESULT_DIR, "evaluation")
if not os.path.exists(PATH_EVALUATION_DIR):
    os.makedirs(PATH_EVALUATION_DIR)

EVAL_TEST_SUMMARY_K = [5, 10, 20, 30, 40, 50]

FILE_EVALUATION_ARTICLE_STATISTICS = os.path.join(PATH_EVALUATION_DIR, "0_article_statistics.json")
FILE_EVALUATION_CLUSTERING_MANUAL = os.path.join(PATH_EVALUATION_DIR, "1_titles_for_clustering.txt")

FILE_EVALUATION_ARTICLE_WIKI_CLASSIFICATION_QUALITY = os.path.join(PATH_EVALUATION_DIR, "4_wiki_classification_quality.json")
FILE_EVALUATION_ARTICLE_CLASSIFICATION_STATISTICS = os.path.join(PATH_EVALUATION_DIR, "4_article_classification_statistics.json")
FILE_EVALUATION_ARTICLE_CLASSIFICATION_MANUAL = os.path.join(PATH_EVALUATION_DIR, "4_article_classification.tsv")
FILE_EVALUATION_ARTICLE_CLASSIFICATION_MANUAL_BALANCED = os.path.join(PATH_EVALUATION_DIR, "4_article_classification_balanced.tsv")

FILE_EVALUATION_ARTICLE_SNIPPET_TRANSLATION = os.path.join(PATH_EVALUATION_DIR, "7_article_snippet_translation.tsv")

EVAL_SUMMARIES_EXPORT_K = 10
PATH_EVALUATION_SUMMARIES_MANUAL = os.path.join(PATH_EVALUATION_DIR, "summaries_manual")
if not os.path.exists(PATH_EVALUATION_SUMMARIES_MANUAL):
    os.makedirs(PATH_EVALUATION_SUMMARIES_MANUAL)

FILE_EVALUATION_SUMMARIES_MANUAL = os.path.join(PATH_EVALUATION_SUMMARIES_MANUAL, "summaries.tsv")


WIKIPEDIA_INFOBOX_INDICATION = ["gbdat", "geboortedatum", "geboren", "sterfdatum", "gbplaats", "geboorteplaats",
                                "sterfplaats", "nationaliteit", "beroep", "titulatuur"]
WIKIPEDIA_OCCUPATIONS_PATH = os.path.join(CACHE_DIR, "categories.json")
WIKIPEDIA_DUMP_PATH = os.path.join(RESOURCE_DIR, "nlwiki-latest-pages-articles.xml")
