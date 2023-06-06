import json

from tqdm import tqdm

from nnsummary.classification.language_model import LanguageModelTrainer
from nnsummary.config import LM_TO_TRAIN
from nnsummary.wikipedia.aspect_mining import AspectMining


def main() -> int:
    print('--' * 60)
    mining = AspectMining()

    person_classes = mining.get_relevant_person_classes()
    print(person_classes)
    print()

    trainer = LanguageModelTrainer()
    results = {}
    for model_name in LM_TO_TRAIN:
        if model_name not in results:
            results[model_name] = {}
        for person in tqdm(person_classes):
            if person not in results[model_name]:
                results[model_name][person] = {}

            aspects = mining.get_person_aspects_for_class(person)
            print(f'training {person} on {aspects}...')
            print('--' * 50)
            results[model_name][person] = trainer.train_language_model_hs_search(aspects, person, model_name,
                                                                                 verbose=True)
            print('--' * 50)

    print('Storing results...')
    with open('results_multiclass_language_models.json', 'wt') as f:
        json.dump(results, f)
    print('Finished (best models are available in models)')
    return 0


if __name__ == '__main__':
    main()
