import krippendorff
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

r1_syntax = ["good", "bad", "bad", "good", "moderate", "bad", "bad", "good", "moderate", "good", "bad", "good", "good",
             "moderate", "good", "moderate", "good", "good", "bad", "moderate", "good", "moderate", "moderate", "good",
             "good", "bad", "moderate", "bad", "moderate", "good", "good", "good", "good", "bad", "moderate", "good",
             "good", "good", "good", "good", "good", "moderate", "good", "good", "good", "good", "moderate", "moderate",
             "good", "moderate", "moderate", "bad", "moderate", "moderate", "moderate", "good", "good", "good",
             "moderate", "moderate", "bad", "bad", "moderate", "bad", "moderate", "bad", "bad", "good", "good", "bad",
             "good", "moderate", "bad", "moderate", "good", "good", "good", "good", "good", "good", "bad", "good",
             "good", "moderate", "good", "good", "good", "good", "good", "moderate", "good", "good", "moderate", "good",
             "moderate", "good", "good", "good", "good", "good"]
r1_syntax_gm = list([r.replace('moderate', 'good') for r in r1_syntax])
r2_syntax = ["good", "bad", "bad", "moderate", "bad", "bad", "bad", "good", "moderate", "moderate", "moderate", "good",
             "good", "moderate", "moderate", "good", "moderate", "moderate", "bad", "moderate", "good", "moderate",
             "moderate", "moderate", "moderate", "moderate", "good", "moderate", "moderate", "moderate", "good",
             "moderate", "good", "bad", "moderate", "moderate", "good", "good", "good", "good", "good", "moderate",
             "good", "good", "good", "good", "moderate", "good", "good", "good", "moderate", "moderate", "moderate",
             "moderate", "bad", "good", "good", "good", "moderate", "good", "bad", "bad", "moderate", "bad", "moderate",
             "good", "bad", "moderate", "good", "bad", "good", "good", "bad", "moderate", "moderate", "good",
             "moderate", "good", "moderate", "moderate", "bad", "moderate", "moderate", "moderate", "moderate",
             "moderate", "good", "moderate", "moderate", "moderate", "moderate", "good", "moderate", "good", "good",
             "moderate", "good", "moderate", "good", "good"]
r2_syntax_gm = list([r.replace('moderate', 'good') for r in r2_syntax])

r1_factual = ["correct", "incorrect", "incorrect", "correct", "incorrect", "correct", "incorrect", "correct", "correct",
              "correct", "incorrect", "incorrect", "correct", "incorrect", "correct", "correct", "correct", "correct",
              "incorrect", "correct", "correct", "correct", "incorrect", "correct", "incorrect", "incorrect", "correct",
              "incorrect", "incorrect", "correct", "correct", "correct", "correct", "correct", "correct", "correct",
              "correct", "correct", "correct", "correct", "correct", "incorrect", "correct", "correct", "incorrect",
              "incorrect", "correct", "incorrect", "correct", "incorrect", "incorrect", "incorrect", "incorrect",
              "correct", "incorrect", "correct", "correct", "incorrect", "correct", "correct", "correct", "incorrect",
              "correct", "incorrect", "incorrect", "incorrect", "incorrect", "correct", "incorrect", "incorrect",
              "correct", "correct", "correct", "correct", "correct", "correct", "correct", "incorrect", "correct",
              "correct", "incorrect", "incorrect", "incorrect", "correct", "correct", "correct", "correct", "incorrect",
              "incorrect", "incorrect", "correct", "incorrect", "correct", "correct", "correct", "correct", "incorrect",
              "correct", "correct", "correct"]
r2_factual = ["correct", "incorrect", "incorrect", "correct", "incorrect", "correct", "incorrect", "correct", "correct",
              "correct", "incorrect", "incorrect", "correct", "incorrect", "incorrect", "correct", "correct", "correct",
              "incorrect", "correct", "correct", "correct", "correct", "correct", "correct", "incorrect", "correct",
              "incorrect", "incorrect", "correct", "correct", "correct", "correct", "correct", "correct", "correct",
              "correct", "correct", "correct", "correct", "correct", "incorrect", "correct", "correct", "correct",
              "correct", "correct", "correct", "correct", "incorrect", "incorrect", "incorrect", "incorrect", "correct",
              "incorrect", "correct", "correct", "incorrect", "correct", "correct", "correct", "correct", "correct",
              "incorrect", "incorrect", "incorrect", "incorrect", "correct", "incorrect", "incorrect", "correct",
              "correct", "correct", "correct", "correct", "correct", "correct", "incorrect", "correct", "correct",
              "incorrect", "incorrect", "incorrect", "correct", "correct", "correct", "correct", "incorrect",
              "incorrect", "incorrect", "correct", "incorrect", "correct", "correct", "correct", "correct", "incorrect",
              "correct", "correct", "correct"]


def count_if(ratings, score):
    return len([1 for r in ratings if r == score])


def get_score(val):
    if val == "good":
        return 2
    if val == "moderate":
        return 1
    if val == "bad":
        return 0

    raise ValueError(f'Value {val} not supported')


def get_score_factual(val):
    if val == "correct":
        return 1
    if val == "incorrect":
        return 0

    raise ValueError(f'Value {val} not supported')


def main() -> int:
    print('Syntax')
    print('-' * 60)
    print(
        f'R1: {count_if(r1_syntax, "good")} x "good" and {count_if(r1_syntax, "moderate")} x "moderate" and  {count_if(r1_syntax, "bad")} x "bad"')
    print(
        f'R2: {count_if(r2_syntax, "good")} x "good" and {count_if(r2_syntax, "moderate")} x "moderate" and  {count_if(r2_syntax, "bad")} x "bad"')

    print('-' * 60)
    r1_syntax_sc = [get_score(r) for r in r1_syntax]
    r2_syntax_sc = [get_score(r) for r in r2_syntax]
    k = krippendorff.alpha(reliability_data=np.array([r1_syntax_sc, r2_syntax_sc]))
    print(f'Krippendorffs Alpha: {k}')

    print('-' * 60)
    table, cats = aggregate_raters(data=np.array([r1_syntax, r2_syntax]).transpose())
    kappa = fleiss_kappa(table, method='fleiss')
    print(f'Fleiss Kapa: {kappa}')

    print('=' * 60)
    print('=' * 60)

    print('-' * 60)
    print(
        f'R1 + good/moderate: {count_if(r1_syntax_gm, "good")} x "good+moderate" and  {count_if(r1_syntax_gm, "bad")} x "bad"')
    print(
        f'R2 + good/moderate: {count_if(r2_syntax_gm, "good")} x "good+moderate" and  {count_if(r2_syntax_gm, "bad")} x "bad"')

    print('-' * 60)
    r1_syntax_sc = [get_score(r) for r in r1_syntax_gm]
    r2_syntax_sc = [get_score(r) for r in r2_syntax_gm]
    k = krippendorff.alpha(reliability_data=np.array([r1_syntax_sc, r2_syntax_sc]))
    print(f'Krippendorffs Alpha: {k}')

    print('-' * 60)
    table, cats = aggregate_raters(data=np.array([r1_syntax_gm, r2_syntax_gm]).transpose())
    kappa = fleiss_kappa(table, method='fleiss')
    print(f'Fleiss Kapa Good+Moderate: {kappa}')

    print('\n\n\n')
    print('Factual correctness')
    print('-' * 60)
    print(f'R1: {count_if(r1_factual, "correct")} x "correct"  and  {count_if(r1_factual, "incorrect")} x "incorrect"')
    print(f'R2: {count_if(r2_factual, "correct")} x "correct"  and  {count_if(r2_factual, "incorrect")} x "incorrect"')

    print('-' * 60)
    r1_factual_sc = [get_score_factual(r) for r in r1_factual]
    r2_factual_sc = [get_score_factual(r) for r in r2_factual]
    k = krippendorff.alpha(reliability_data=np.array([r1_factual_sc, r2_factual_sc]))
    print(f'Krippendorffs Alpha: {k}')

    print('-' * 60)
    table, cats = aggregate_raters(data=np.array([r1_factual, r2_factual]).transpose())
    kappa = fleiss_kappa(table, method='fleiss')
    print(f'Fleiss Kapa: {kappa}')

    return 0


if __name__ == '__main__':
    main()
