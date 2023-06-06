import os.path
import pickle
import random

import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from nnsummary.config import CLUSTERING_SAMPLE_SIZE, CLUSTERING_SECTION_SIMILARITY, \
    FILE_CACHE_WIKIPEDIA_SECTION_CLUSTERING
from nnsummary.wikipedia.people_processor import WikipediaPeopleProcessor


class WikipediaSectionClustering:

    def __init__(self):
        self.model = SentenceTransformer('jegormeister/bert-base-dutch-cased')
        self.model_dim = 768
        self.clustering_results = None
        self.clustered_section_titles = set()
        if os.path.isfile(FILE_CACHE_WIKIPEDIA_SECTION_CLUSTERING):
            print(f'Loading clustering results from {FILE_CACHE_WIKIPEDIA_SECTION_CLUSTERING}...')
            with open(FILE_CACHE_WIKIPEDIA_SECTION_CLUSTERING, 'rb') as f:
                self.clustering_results, self.clustered_section_titles = pickle.load(f)
        else:
            print('Computing section title clustering...')
            self.wiki = WikipediaPeopleProcessor()
            self.__apply_clustering()
            print(f'Writing results to {FILE_CACHE_WIKIPEDIA_SECTION_CLUSTERING}...')
            with open(FILE_CACHE_WIKIPEDIA_SECTION_CLUSTERING, 'wb') as f:
                pickle.dump((self.clustering_results, self.clustered_section_titles), f)

    def __apply_clustering(self):
        """
        Performs the initial clustering step
        1. average the sentence vectors of a section
        2. average the vectors of all vectors of a section title
        3. compare those vectors
        :return:
        """
        section2texts_frequent = {k: v for k, v in self.wiki.section2texts.items() if len(v) >= CLUSTERING_SAMPLE_SIZE}

        centroids = []
        for aspect, texts in tqdm(section2texts_frequent.items()):
            texts = random.sample(texts, CLUSTERING_SAMPLE_SIZE)
            text_embeddings = np.ndarray((len(texts), self.model_dim))
            for idx, t in enumerate(texts):
                sentences = []
                for s in nltk.sent_tokenize(t):
                    sentences.append(s.strip())

                    embeddings = self.model.encode(sentences)
                    # compute the mean along the main axis
                    text_centroid = embeddings.mean(0)
                    text_embeddings[idx] = text_centroid

            # compute the mean along the main axis
            centroid = text_embeddings.mean(0)
            centroids.append((aspect, centroid))

        clustering_results = []
        for a1, c1 in centroids:
            for a2, c2 in centroids:
                if a1 == a2:
                    continue

                cosine_sim = float(util.cos_sim(c1, c2)[0][0])
                if cosine_sim >= CLUSTERING_SECTION_SIMILARITY:
                    clustering_results.append((a1, a2, cosine_sim))

        clustering_results.sort(key=lambda x: x[2], reverse=True)
        for a1, a2, cosine_sim in clustering_results[:10]:
            print(f'{round(cosine_sim, 3)}: {a1} <=> {a2} ')

        print()
        print(f'Found {len(clustering_results)} more results...')
        self.clustering_results = clustering_results
        self.__compute_clusters()

    def __compute_clusters(self):
        """
        Computes the actual set of clusters, i.e. the sets of section titles that belong together
        :return:
        """
        merged_aspects = {}
        # initial setup
        for a1, a2, cosine_sim in self.clustering_results:
            if a1 not in merged_aspects:
                merged_aspects[a1] = set()
            merged_aspects[a1].add(a2)

        # compute transitive closure
        changed = True
        iterations = 0
        while changed:
            iterations += 1
            changed = False
            merged_aspects_new = {}
            for aspect, mapped_to in merged_aspects.items():
                mapped_to_new = mapped_to.copy()
                for asp_mapped in mapped_to:
                    if asp_mapped in merged_aspects:
                        # Also add all transitive connections
                        transitive_elements = merged_aspects[asp_mapped]
                        if len(transitive_elements) != len(transitive_elements.intersection(mapped_to_new)):
                            mapped_to_new.update(transitive_elements)
                            changed = True

                merged_aspects_new[aspect] = mapped_to_new

            merged_aspects = merged_aspects_new

        print(f'Stable after {iterations} iterations')
        # for k, v in merged_aspects.items():
        #    print(f'{k} => {v}\n')
        print(f'Unifying {len(merged_aspects)} sets...')
        self.clustered_section_titles = []
        for k, v in merged_aspects.items():
            candidate_set = set()
            candidate_set.add(k)
            candidate_set.update(v)

            duplicate_found = False
            for stable_set in self.clustered_section_titles:
                if len(candidate_set.union(stable_set)) == len(candidate_set.intersection(stable_set)):
                    # we have the set already - stop
                    duplicate_found = True
                    break
            if not duplicate_found:
                self.clustered_section_titles.append(candidate_set)

        print(f'Found {len(self.clustered_section_titles)} stable sets...')
        print()
        for stable_set in self.clustered_section_titles:
            print(stable_set)

    def find_also_name_as_titles(self, title):
        for cluster in self.clustered_section_titles:
            if title in cluster:
                alt_names = sorted(cluster)
                return ' | '.join([a for a in alt_names])
        return title


def main() -> int:
    clustering = WikipediaSectionClustering()
    for cluster in clustering.clustered_section_titles:
        print(cluster)

    print('--' * 60)
    print(clustering.find_also_name_as_titles("no int"))
    print(clustering.find_also_name_as_titles("werk"))
    print(clustering.find_also_name_as_titles("latere leven"))
    return 0


if __name__ == '__main__':
    main()
