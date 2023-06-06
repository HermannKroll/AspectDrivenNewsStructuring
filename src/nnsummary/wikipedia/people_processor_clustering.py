from nnsummary.wikipedia.clustering import WikipediaSectionClustering
from nnsummary.wikipedia.people_processor import WikipediaPeopleProcessor


class WikipediaPeopleProcessorWithClustering:

    def __init__(self):
        self.wiki = WikipediaPeopleProcessor()
        self.clustering = WikipediaSectionClustering()

        self.person2titles = {}
        self.section2count = {}
        self.person_title2text = {}
        self.person2titles = {}
        self.remap_section_titles_based_on_clustering()

    def remap_section_titles_based_on_clustering(self):
        title_mapping_plan = {}
        for stable_set in self.clustering.clustered_section_titles:
            sorted_names = sorted(stable_set)
            map_to = sorted_names[0]
            for s in sorted_names[1:]:
                title_mapping_plan[s] = map_to

        print(f'Will map {len(title_mapping_plan)} section titles...')
        print('Mapping person2titles dictionary...')
        self.person2titles = {}
        for person, titles in self.wiki.person2titles.items():
            self.person2titles[person] = set()
            for t in titles:
                if t in title_mapping_plan:
                    self.person2titles[person].add(title_mapping_plan[t])
                else:
                    self.person2titles[person].add(t)

        print('Mapping section2count dictionary...')
        self.section2count = {}
        for t, count in self.wiki.section2count.items():
            title_to_add = t
            if t in title_mapping_plan:
                title_to_add = title_mapping_plan[t]

            if title_to_add in self.section2count:
                self.section2count[title_to_add] += count
            else:
                self.section2count[title_to_add] = count

        print('Mapping person_title2text dictionary...')
        for k, text in self.wiki.person_title2text.items():
            p, title = k
            title_to_add = title
            if title in title_mapping_plan:
                title_to_add = title_mapping_plan[title]

            if (p, title_to_add) in self.person_title2text:
                self.person_title2text[(p, title_to_add)] += text
            else:
                self.person_title2text[(p, title_to_add)] = text

        print('Mapping person2title dictionary...')
        print(f'Starting with {len(self.wiki.person2titles)} persons...')
        self.person2titles = {}
        for p, titles in self.wiki.person2titles.items():
            self.person2titles[p] = set()
            for title in titles:
                title_to_add = title
                if title in title_mapping_plan:
                    title_to_add = title_mapping_plan[title]

                self.person2titles[p].add(title_to_add)

        print(f'Received {len(self.person2titles)} persons...')

    def print_section2count_info(self):
        print('OLD')
        section_count = list([(k, v) for k, v in self.wiki.section2count.items()])
        section_count.sort(key=lambda x: x[1], reverse=True)

        print(section_count[:150])
        print()

        print('Remapped NEW')
        print('==' * 60)
        print()
        section_count = list([(k, v) for k, v in self.section2count.items()])
        section_count.sort(key=lambda x: x[1], reverse=True)

        print(section_count[:150])


def main() -> int:
    wiki = WikipediaPeopleProcessorWithClustering()
    wiki.print_section2count_info()
    print(f'{len(wiki.clustering.clustered_section_titles)} clusters were computed')
    print('Based on the following clusters...')
    for s in wiki.clustering.clustered_section_titles:
        print(s)
    return 0


if __name__ == '__main__':
    main()
