import krippendorff
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

r1 = [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1,
      1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1,
      1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
r2 = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
      1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
      1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0]
r3 = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
      1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def majority_vote(a, b, c):
    if a == b:
        return a
    if a == c:
        return a
    if b == c:
        return b


def count_if(ratings, score):
    return len([1 for r in ratings if r == score])


def main() -> int:
    print(f'R1: {count_if(r1, 1)} x "1" and {count_if(r1, 0)} x 0')
    print(f'R2: {count_if(r2, 1)} x "1" and {count_if(r2, 0)} x 0')
    print(f'R3: {count_if(r3, 1)} x "1" and {count_if(r3, 0)} x 0')

    majority = []
    for a, b, c in zip(r1, r2, r3):
        majority.append(majority_vote(a, b, c))

    print('-'*60)
    print(f'Majority: {count_if(majority, 1)} x "1" and {count_if(majority, 0)} x 0')
    print('-'*60)

    k = krippendorff.alpha(reliability_data=np.array([r1, r2, r3]))
    print(f'Krippendorffs Alpha: {k}')

    table, cats = aggregate_raters(data=np.array([r1, r2, r3]).transpose())
    kappa = fleiss_kappa(table, method='fleiss')
    print(f'Fleiss Kapa: {kappa}')
    return 0


if __name__ == '__main__':
    main()
