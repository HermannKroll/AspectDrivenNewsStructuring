import re
import pandas as pd
import numpy as np
import textstat
import os
import json
# from easse.quality_estimation import corpus_quality_estimation

from typing import List
from tseval.feature_extraction import (
    get_compression_ratio,
    count_sentence_splits,
    get_levenshtein_similarity,
    is_exact_match,
    get_additions_proportion,
    get_deletions_proportion,
    get_wordrank_score,
    wrap_single_sentence_vectorizer,
)

from easse.utils.preprocessing import normalize


def get_average(vectorizer, orig_sentences, sys_sentences):
    cumsum = 0
    count = 0
    value_list = []
    for orig_sentence, sys_sentence in zip(orig_sentences, sys_sentences):
        value = vectorizer(orig_sentence, sys_sentence)
        cumsum += value
        value_list.append(value)
        count += 1
    return_string = "{mean:.2f} ± {std:.1f}".format(mean=cumsum / count, std=np.std(value_list))
    return return_string


def corpus_quality_estimation(
        orig_sentences: List[str], sys_sentences: List[str], lowercase: bool = False, tokenizer: str = '13a'
):
    orig_sentences = [normalize(sent, lowercase, tokenizer) for sent in orig_sentences]
    sys_sentences = [normalize(sent, lowercase, tokenizer) for sent in sys_sentences]
    return {
        'Compression ratio': get_average(get_compression_ratio, orig_sentences, sys_sentences),
        'Sentence splits': get_average(count_sentence_splits, orig_sentences, sys_sentences),
        'Levenshtein similarity': get_average(get_levenshtein_similarity, orig_sentences, sys_sentences),
        'Exact copies': get_average(is_exact_match, orig_sentences, sys_sentences),
        'Additions proportion': get_average(get_additions_proportion, orig_sentences, sys_sentences),
        'Deletions proportion': get_average(get_deletions_proportion, orig_sentences, sys_sentences),
        'Lexical complexity score': get_average(
            wrap_single_sentence_vectorizer(get_wordrank_score), orig_sentences, sys_sentences
        ),
    }


def read_in_datasets():
    runs = {}

    for k in ['5', '10', '20', '30', '40', '50']:
        runs[k] = {'src': [], 'simp': [], 'simp_nl': []}

    for filename in os.listdir('../../../cache/eval summaries/'):
        file = os.path.join('../../../cache/eval summaries/', filename)
        if '5_eval_summaries_' in file:
            # checking if it is a file
            if os.path.isfile(file):
                print(file)
                with open(file) as f:

                    person_json = json.load(f)

                    for role in person_json['roles']:
                        for summary_k in role['summary']:
                            for k in summary_k['summaries_k']:
                                runs[k]['simp'].append(summary_k['summaries_k'][k]['summary'])
                                runs[k]['src'].append(summary_k['summaries_k'][k]['summary_input_text'])
                                runs[k]['simp_nl'].append(summary_k['summaries_k'][k]['summary_nl'])

    return runs


def calc_quality_estimation(s, p):
    p_r_f1 = corpus_quality_estimation(orig_sentences=s, sys_sentences=p)
    return p_r_f1


def calc_readability(dataset, nl_dataset):
    readability = {'flesch': [],
                   # 'flesch_kincaid': [],
                   # 'coleman_liau': [],
                   # 'automated_readability': [],
                   # 'linsear_write': [],
                   # 'gunning_fog': [],
                   'dale_chall': [],
                   'difficult_words': [],
                   'text_standard': [],
                   'reading_time': [],
                   'syllables': [],
                   'lexicon': [],
                   'sentences': [],
                   'flesch_nl': []}

    textstat.set_lang('en')

    for text in dataset:
        # grade levels by magic formulas
        # readability['flesch_kincaid'].append(textstat.flesch_kincaid_grade(text))
        # readability['coleman_liau'].append(textstat.coleman_liau_index(text))
        # readability['automated_readability'].append(textstat.automated_readability_index(text))
        # readability['gunning_fog'].append(textstat.gunning_fog(text))
        # grade score for first 100 words: (easy words+(difficult words * 3))/sentences
        # readability['linsear_write'].append(textstat.linsear_write_formula(text))

        # readability score, 30-49: difficult, 50-59: fairly difficult, 60-69: standard
        readability['flesch'].append(textstat.flesch_reading_ease(text))

        # grade level with 3000 most common English words lookup,
        # <4.9: 4th grade or lower, 5-5.9: 5/6th, 6-6.9:7/8th, 7-7.9: 9/10th, 8-8.9: 11/12th, 9-9.9: 13-15th (college student)
        readability['dale_chall'].append(textstat.dale_chall_readability_score(text))

        # number words with >= 3 syllables and which are not in list of easy words
        readability['difficult_words'].append(textstat.difficult_words(text))

        # readability consensus
        readability['text_standard'].append(int(re.search(r'\d+', textstat.text_standard(text)).group()))

        # seconds to read text
        readability['reading_time'].append(textstat.reading_time(text, ms_per_char=14.69))

        # number of syllabes
        readability['syllables'].append(textstat.syllable_count(text))
        # number of different words
        readability['lexicon'].append(textstat.lexicon_count(text))
        # number of sentences
        readability['sentences'].append(textstat.sentence_count(text))

    textstat.set_lang('nl')

    for text in nl_dataset:
        # readability score, 30-49: difficult, 50-59: fairly difficult, 60-69: standard
        readability['flesch_nl'].append(textstat.flesch_reading_ease(text))

    for measure in readability:
        readability[measure] = "{mean:.2f} ± {std:.1f}".format(
            mean=sum(readability[measure]) / len(readability[measure]), std=np.std(readability[measure]))

    return readability


def main():
    """## Generate Scores"""

    runs = read_in_datasets()
    all_results = {}
    for r in runs:
        scores = calc_readability(runs[r]['simp'], runs[r]['simp_nl'])
        # print(scores)
        scores.update(calc_quality_estimation(runs[r]['src'], runs[r]['simp']))
        # print(scores)
        # print(calc_quality_estimation(runs[r]['src'], runs[r]['simp']))
        all_results[r] = scores

    df = pd.DataFrame.from_dict(all_results)

    # print(df)
    print(df.to_latex(index=False, formatters={"name": str.upper}, float_format="{:.1f}".format))


if __name__ == '__main__':
    main()
