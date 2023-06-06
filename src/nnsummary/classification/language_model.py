import itertools
import json
import random
import evaluate
import numpy as np
import torch
import os
import shutil

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from nnsummary.config import LM_WEIGHT_DECAYS, LM_NUM_TRAIN_EPOCHS, LM_LEARNING_RATES, LM_TRAIN_RATIO, \
    SECTION_MIN_LENGTH, PATH_LM_MODEL_OUTPUT_DIR, LM_VAL_TEST_RATIO, LM_RANDOM_SEED, \
    ASPECT_MINING_MINIMUM_TEXTS, PATH_LM_MODEL_TRAIN_STATISTICS
from nnsummary.wikipedia.people_processor_clustering import WikipediaPeopleProcessorWithClustering

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.utils import logging

logging.set_verbosity_error()


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class SampledData:

    def __init__(self, x_pos_train, x_pos_val, x_pos_test, y_pos_train, y_pos_val, y_pos_test,
                 x_neg_train, x_neg_val, x_neg_test, y_neg_train, y_neg_val, y_neg_test):
        self.x_pos_train = x_pos_train
        self.x_pos_val = x_pos_val
        self.x_pos_test = x_pos_test
        self.y_pos_train = y_pos_train
        self.y_pos_val = y_pos_val
        self.y_pos_test = y_pos_test

        self.x_neg_train = x_neg_train
        self.x_neg_val = x_neg_val
        self.x_neg_test = x_neg_test
        self.y_neg_train = y_neg_train
        self.y_neg_val = y_neg_val
        self.y_neg_test = y_neg_test

    def get_train_size(self):
        return len(self.x_pos_train) + len(self.x_neg_train)


class LanguageModelTrainer:

    def __init__(self):
        self.wiki = WikipediaPeopleProcessorWithClustering()

        self.tokenizer = None
        self.num_labels = None
        self.id2label = None
        self.label2id = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.y_val = None
        self.y_test = None

    # Get texts for one category
    def get_positive_examples_multiclass(self, aspect, category):
        texts = [self.wiki.person_title2text[(p, aspect)] for p in self.wiki.wiki.category2person[category] if
                 (p, aspect) in self.wiki.person_title2text]
        if len([t for t in texts if len(t) < ASPECT_MINING_MINIMUM_TEXTS]) > 0:
            raise ValueError(f'found some training text less than {ASPECT_MINING_MINIMUM_TEXTS} characters')
        return texts

    def get_negative_examples_multiclass(self, aspects, category, sample_size):
        keys_to_ignore = set()
        for aspect in aspects:
            for p in self.wiki.wiki.category2person[category]:
                keys_to_ignore.add(f'{p}___{aspect}')

        # print('ignoree ', keys_to_ignore)
        texts = []
        for k, text in self.wiki.person_title2text.items():
            key = f'{k[0]}___{k[1]}'
            if key in keys_to_ignore:
                # print('ignore ', key)
                continue
            if len(text) < SECTION_MIN_LENGTH:
                raise ValueError('found some training text less than 100 characters')

            texts.append(text)
        return random.sample(texts, sample_size)

    def train_language_model(self, model_name, params):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        accuracy = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_labels, id2label=self.id2label, label2id=self.label2id
        )

        model_name_fixed = model_name.replace('/', '-')
        output_dir = os.path.join(PATH_LM_MODEL_OUTPUT_DIR, f'tmp_{model_name_fixed}')
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=params["num_train_epochs"],
            weight_decay=params["weight_decay"],
            evaluation_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        result = trainer.evaluate(self.val_dataset)

        y_pred = trainer.predict(self.val_dataset)
        y_pred = np.argmax(y_pred.predictions, axis=-1)
        result["macro_precision"] = precision_score(self.y_val, y_pred, average='macro')
        result["macro_recall"] = recall_score(self.y_val, y_pred, average='macro')
        result["macro_f1"] = f1_score(self.y_val, y_pred, average='macro')

        return result, trainer

    def __print_details_on_set(self, name, x, y):
        count = [0] * (1 + max([e for e in y]))
        for entry in y:
            count[entry] += 1
        print(f'{name} has {len(x)} entries (labels neg = {count[0]} / pos = {count[1:]})')

    def random_sample_training_data(self, aspects, person, verbose=True):
        result = {"aspects": {}}
        result["aspects"]["all"] = aspects
        positive_texts_per_aspect = []
        for idx, aspect in enumerate(aspects):
            pos_texts_aspect = self.get_positive_examples_multiclass(aspect, person)
            result["aspects"][aspect] = len(pos_texts_aspect)
            if verbose:
                print(aspect, len(pos_texts_aspect))
            positive_texts_per_aspect.append(pos_texts_aspect)

        min_example_size = min([len(t) for t in positive_texts_per_aspect])
        if verbose:
            print(f'Sampling down to {min_example_size}')
        result["aspects"]["balancing_to"] = min_example_size

        # Balance all examples
        x_pos_train = []
        x_pos_val = []
        x_pos_test = []
        y_pos_train = []
        y_pos_val = []
        y_pos_test = []
        positive_labels_size = []
        no_pos_examples = 0
        for idx, texts in enumerate(positive_texts_per_aspect):
            if len(texts) > min_example_size:
                pos_texts_for_aspect = random.sample(texts, min_example_size)
            elif len(texts) == min_example_size:
                pos_texts_for_aspect = texts
            else:
                raise ValueError('This should not happen - Data must be balanced')

            no_pos_examples += len(pos_texts_for_aspect)
            pos_aspect_labels = list([(idx + 1)] * min_example_size)
            xp_train, xp_test, yp_train, yp_test = train_test_split(pos_texts_for_aspect, pos_aspect_labels,
                                                                    test_size=1 - LM_TRAIN_RATIO,
                                                                    random_state=LM_RANDOM_SEED)

            # split test again in 50 - 50
            xp_test, xp_val, yp_test, yp_val = train_test_split(xp_test, yp_test, test_size=LM_VAL_TEST_RATIO,
                                                                random_state=LM_RANDOM_SEED)

            x_pos_train.extend(xp_train)
            x_pos_val.extend(xp_val)
            x_pos_test.extend(xp_test)

            y_pos_train.extend(yp_train)
            y_pos_val.extend(yp_val)
            y_pos_test.extend(yp_test)
            # train_positive_texts.extend(xp_train)
            # positive_labels_size.append(len(aspect_labels))
            # positive_labels.extend(aspect_labels)

        random.seed(LM_RANDOM_SEED)
        neg_sample_size = no_pos_examples
        # Carefully consider all aspects that should not be selected
        negative_texts = self.get_negative_examples_multiclass(aspects, person, neg_sample_size)
        negative_labels = list([0 for i in range(len(negative_texts))])

        result["aspects"]["examples_pos"] = no_pos_examples
        result["aspects"]["examples_neg"] = len(negative_texts)

        x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(negative_texts, negative_labels,
                                                                            test_size=1 - LM_TRAIN_RATIO,
                                                                            random_state=LM_RANDOM_SEED)

        # split test again in 50 - 50
        x_neg_test, x_neg_val, y_neg_test, y_neg_val = train_test_split(x_neg_test, y_neg_test,
                                                                        test_size=LM_VAL_TEST_RATIO,
                                                                        random_state=LM_RANDOM_SEED)

        data = SampledData(x_pos_train=x_pos_train, x_pos_val=x_pos_val, x_pos_test=x_pos_test,
                           y_pos_train=y_pos_train, y_pos_val=y_pos_val, y_pos_test=y_pos_test,
                           x_neg_train=x_neg_train, x_neg_val=x_neg_val, x_neg_test=x_neg_test,
                           y_neg_train=y_neg_train, y_neg_val=y_neg_val, y_neg_test=y_neg_test)

        return data

    def __generate_datasets(self, aspects, person):
        for idx, aspect in enumerate(aspects):
            self.id2label[idx + 1] = aspect
            self.label2id[aspect] = idx + 1

        self.num_labels = len(self.id2label)

        data = self.random_sample_training_data(aspects, person)

        # Compute the final tests
        x_train, y_train = (data.x_pos_train + data.x_neg_train), (data.y_pos_train + data.y_neg_train)
        x_val, self.y_val = (data.x_pos_val + data.x_neg_val), (data.y_pos_val + data.y_neg_val)
        x_test, self.y_test = (data.x_pos_test + data.x_neg_test), (data.y_pos_test + data.y_neg_test)

        print('--' * 60)
        self.__print_details_on_set("train", x_train, y_train)
        self.__print_details_on_set("test", x_test, self.y_test)
        self.__print_details_on_set("val", x_val, self.y_val)
        print('--' * 60)

        train_encodings = self.tokenizer(x_train, truncation=True, padding=True)
        test_encodings = self.tokenizer(x_test, truncation=True, padding=True)
        val_encodings = self.tokenizer(x_val, truncation=True, padding=True)

        self.train_dataset = NewsDataset(train_encodings, y_train)
        self.test_dataset = NewsDataset(test_encodings, self.y_test)
        self.val_dataset = NewsDataset(val_encodings, self.y_val)

        return result

        # x_data, y_data = (positive_texts + negative_texts), (positive_labels + negative_labels)

    # if verbose: print(
    #     f'Retrieved {len(x_data)} texts (and {positive_labels_size} pos + {len(negative_labels)} neg labels)')

    # First split train and test 80 - 20
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - LM_TRAIN_RATIO,
    #                                                  random_state=LM_RANDOM_SEED)
    # split test again in 50 - 50
    # x_test, x_val, y_test, self.y_val = train_test_split(x_test, y_test, test_size=LM_VAL_TEST_RATIO,
    #                                                    random_state=LM_RANDOM_SEED)

    def train_language_model_hs_search(self, aspects, person, model_name, verbose=True):
        best_precision_macro = 0
        best_results = None
        model_name_path = model_name.replace('/', '-')
        output_dir = os.path.join(PATH_LM_MODEL_OUTPUT_DIR, f'{model_name_path}_{person}')

        if os.path.isdir(output_dir):
            print(f'Skipping - model already trained ({output_dir})')
            return

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.id2label = {0: "NEGATIVE"}
        self.label2id = {"NEGATIVE": 0}

        aspects = sorted(aspects)
        self.__generate_datasets(aspects, person)

        for param_comb in itertools.product(LM_LEARNING_RATES, LM_NUM_TRAIN_EPOCHS, LM_WEIGHT_DECAYS):
            print('==' * 60)
            print('==' * 60)
            params = {"learning_rate": param_comb[0],
                      "num_train_epochs": param_comb[1],
                      "weight_decay": param_comb[2]}

            print(f'Searching with params: {params} on {person}')
            print('==' * 60)
            print('==' * 60)
            results, trainer = self.train_language_model(model_name, params)
            if results["macro_precision"] > best_precision_macro or not best_results:
                best_results = {"validation": results, "params": params}
                best_precision_macro = results["macro_precision"]

                test_result = trainer.evaluate(self.test_dataset)
                y_pred = trainer.predict(self.test_dataset)
                y_pred = np.argmax(y_pred.predictions, axis=-1)
                test_result["macro_precision"] = precision_score(self.y_test, y_pred, average='macro')
                test_result["macro_recall"] = recall_score(self.y_test, y_pred, average='macro')
                test_result["macro_f1"] = f1_score(self.y_test, y_pred, average='macro')
                best_results["test"] = test_result

                print(best_results)
                if os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)

                trainer.save_model(output_dir)

        model_name_path = model_name.replace('/', '-')
        model_name_path = os.path.join(PATH_LM_MODEL_TRAIN_STATISTICS, f"results_{model_name_path}_{person}.json")
        with open(model_name_path, 'wt') as f:
            json.dump(best_results, f)
