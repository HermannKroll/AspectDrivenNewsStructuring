import json

from tqdm import tqdm

from nnsummary.config import FILE_WIKIPEDIA_CATEGORIES


def clean_category_name(category_name):
    cat_name = category_name.lower()
    cat_name = cat_name.replace(' naar nationaliteit', '')
    cat_name = cat_name.replace(' naar beroep', '')
    return cat_name.strip()


class CategoryTree:
    def __init__(self, file_path=FILE_WIKIPEDIA_CATEGORIES):
        with open(file_path) as f:
            category_dict = json.load(f)
        self.categories = set()
        self.category2level = {}
        self.category2all_subcategories = {}
        self.category2direct_subcategories = {}
        self.category2all_supercategories = {}
        self.category2direct_supercategories = {}
        self.root = "persoon"
        self._parse_tree(category_dict, self.root)
        self._populate_all()

    def _add_category_to_level(self, category, level):
        if category not in self.category2level:
            self.category2level[category] = level
        else:
            self.category2level[category] = min(level, self.category2level[category])

    def _add_category_child(self, category, child, level):
        self.categories.add(category)
        self.categories.add(child)

        # Add category levels
        self._add_category_to_level(category, level)
        self._add_category_to_level(child, level + 1)

        # Create sub category relation
        if category not in self.category2direct_subcategories:
            self.category2direct_subcategories[category] = set()
        self.category2direct_subcategories[category].add(child)

        # Create super category relation
        if child not in self.category2direct_supercategories:
            self.category2direct_supercategories[child] = set()
            # add direct parent
        self.category2direct_supercategories[child].add(category)

    def _populate_all(self):
        for cat in tqdm(self.categories):
            # Find all transitive sub categories
            self.category2all_subcategories[cat] = set()
            todo = [cat]
            visited = set()
            while todo:
                current = todo.pop()
                visited.add(current)
                if current not in self.category2direct_subcategories:
                    continue
                for sub_cat in self.category2direct_subcategories[current]:
                    self.category2all_subcategories[cat].add(sub_cat)
                    if sub_cat not in visited:
                        todo.append(sub_cat)

            # Find all transitive super categories
            self.category2all_supercategories[cat] = set()
            todo = [cat]
            visited = set()
            while todo:
                current = todo.pop()
                visited.add(current)
                if current not in self.category2direct_supercategories:
                    continue
                for super_cat in self.category2direct_supercategories[current]:
                    self.category2all_supercategories[cat].add(super_cat)
                    if super_cat not in visited:
                        todo.append(super_cat)

    def _parse_tree(self, category_dict, parent, level=0):
        for category, children_dict in category_dict.items():
            category = clean_category_name(category)
            self._add_category_child(parent, category, level)
            if len(children_dict) > 0:
                self._parse_tree(children_dict, parent=category, level=level + 1)

    def find_sub_categories(self, category_name):
        name = clean_category_name(category_name)
        if name in self.category2all_subcategories:
            return self.category2all_subcategories[name]
        else:
            return set()

    def find_direct_sub_categories(self, category_name):
        name = clean_category_name(category_name)
        if name in self.category2direct_subcategories:
            return self.category2direct_subcategories[name]
        else:
            return set()

    def find_super_categories(self, category_name):
        name = clean_category_name(category_name)
        if name in self.category2all_supercategories:
            return self.category2all_supercategories[name]
        else:
            return set()

    def find_direct_super_categories(self, category_name):
        name = clean_category_name(category_name)
        if name in self.category2direct_supercategories:
            return self.category2direct_supercategories[name]
        else:
            return set()

    def filter_occupations_by_level(self, occupations, max_level):
        return {o for o in occupations if o in self.category2level and self.category2level[o] <= max_level}

    def is_known(self, category_name):
        name = clean_category_name(category_name)
        return name in self.categories


def main() -> int:
    cat_tree = CategoryTree()
    print(f'Found {len(cat_tree.categories)} categories')

    super_cats = cat_tree.find_super_categories("persoon naar beroep")
    print(f'super concepts: {super_cats}')

    super_cats = cat_tree.find_super_categories("amerikaanse gouverneursverkiezingen")
    print(f'super concepts: {super_cats}')
    print(cat_tree.filter_occupations_by_level(super_cats, 3))

    print()
    sub_cats = cat_tree.find_sub_categories("natuurbeschermer")
    print(f'sub concepts: {sub_cats}')
    print(cat_tree.filter_occupations_by_level(sub_cats, 3))

    print()
    sub_cats = cat_tree.find_sub_categories("politicus")
    print(f'sub concepts: {len(sub_cats)} for politicus')
    # print(cat_tree.filter_occupations_by_level(sub_cats, 1))
    return 0


if __name__ == '__main__':
    main()
