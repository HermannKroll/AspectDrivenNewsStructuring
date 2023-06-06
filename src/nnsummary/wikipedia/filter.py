import xml.etree.ElementTree as ET
import re
import json
import mwparserfromhell
from nnsummary.config import WIKIPEDIA_INFOBOX_INDICATION, WIKIPEDIA_OCCUPATIONS_PATH, WIKIPEDIA_DUMP_PATH

infobox_regex = re.compile(r'{{Infobox.*?^}}', re.DOTALL | re.MULTILINE)
categories_regex = re.compile(r'\[\[Categorie:.*?\]\]', re.MULTILINE)
title_pattern = re.compile(r'=+\s*(.*?)\s*=+')


def read_list_from_file(filename):

    def convert_dict_to_list(occs):
        result = []

        for key, value in occs.items():
            result.append(key)
            if isinstance(value, dict):
                sub_list = convert_dict_to_list(value)
                result.extend(sub_list)

        return result
    with open(filename) as file:
        occupations_dict = json.load(file)

    occupations = convert_dict_to_list(occupations_dict)

    return occupations


def strip_tag_name(tag):
    t = tag
    idx = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t


def clean_text(text):
    text = re.sub(r"\{\{Infobox\s+\S+([\s\S]*?\n\}\})\n", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{\s*Infobox[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[Categorie:.*?\]\]", "", text)
    text = re.sub(r'\{\|[\s\S]*?\|\}', "", text)
    text = re.sub(r"<ref\b[^>]*>(?:[^<]|<(?!/?ref\b))*?</ref>", "", text)
    text = re.sub(r'^\[\[Bestand:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\[\[File:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\{\| style=.*?\|\}', "", text, flags=re.DOTALL)
    text = re.sub(r'\| colspan=.*?\n\|-----\n', "", text, flags=re.DOTALL)
    text = re.sub(r'\| colspan=.*?\|', "", text, flags=re.DOTALL)
    text = re.sub(r'\| align=.*?\|\}', "", text, flags=re.DOTALL)
    text = re.sub(r'\| width=.*?\]\]', "", text, flags=re.DOTALL)
    text = re.sub(r'\! colspan=.*?\|-----', "", text, flags=re.DOTALL)
    text = re.sub(r'align=.*?\n', "", text, flags=re.DOTALL)
    wikicode = mwparserfromhell.parse(text)
    section_list = list()
    summary = ""
    covered_titles = []
    result = ""
    position = 0
    for idx, section in enumerate(wikicode.get_sections()):
        cleaned_text = section.strip_code().strip()
        title = re.findall(title_pattern, str(section))
        if idx == 0 and not title:
            summary = cleaned_text
        elif len(title) > 1:
            covered_titles += title
            text_parts = []
            current_text = ""
            for i, line in enumerate(str(section).split("\n")):
                match = re.match(title_pattern, line)
                if match and i == 0:
                    current_text += str(match.group(0)) + "\n"
                elif match:
                    text_parts.append(current_text)
                    current_text = str(match.group(0)) + "\n"
                else:
                    current_text += line + "\n"
            text_parts.append(current_text)
            for j, part in enumerate(text_parts):
                section_list.append(
                    {"org_title": str(part).split("\n")[0], "title": str(part).split("\n")[0].strip("="),
                     "position": position,
                     "text": re.sub(title_pattern, '', part).strip()})
                position += 1
        else:
            if title[0] not in covered_titles:
                section_list.append({"org_title": str(section).split("\n")[0], "title": title[0], "position": position,
                                     "text": re.sub(title_pattern, '', cleaned_text).strip()})
                position += 1

        section_text = section.strip_code().strip()
        if section_text:
            result += section_text + "\n"
    return summary, section_list


def filter_wikipedia_dump():
    infobox_indications = WIKIPEDIA_INFOBOX_INDICATION
    revision_counter = 0
    occupation_list = read_list_from_file(WIKIPEDIA_OCCUPATIONS_PATH)
    for event, elem in ET.iterparse(WIKIPEDIA_DUMP_PATH,
                                    events=('start', 'end')):
        tag = strip_tag_name(elem.tag)
        if tag == "revision":
            revision_counter += 1
        if event == 'end':
            if tag == "title":
                title = elem.text
            if tag == "id":
                if revision_counter % 2 == 0:
                    curid = elem.text

            if tag == "text":
                is_person = False
                occupations = []
                try:
                    infobox = re.search(infobox_regex, elem.text)
                except:
                    continue

                if infobox:
                    infobox = infobox.group(0)
                    pattern = r"{{Infobox\s+([\w\s]+)\s+"
                    match_start = re.search(pattern, infobox)
                    if match_start:
                        infobox_type = match_start.group(1)
                        if "persoon" in str(infobox_type).lower():
                            is_person = True
                    infobox_dict = {}
                    for match in re.finditer(r"\|\s*(.*?)\s*=\s*(.*?)\s*(?=\||}})", infobox):
                        key = match.group(1)
                        value = match.group(2)
                        infobox_dict[key] = value
                    if not is_person:
                        dict_keys = set([key.lower() for key in infobox_dict.keys()])
                        list_items = set(infobox_indications)
                        common_elements = dict_keys.intersection(list_items)
                        if common_elements:
                            is_person = True
                    if "beroep" in infobox_dict.keys():
                        if infobox_dict["beroep"] != "":
                            beroep = infobox_dict["beroep"]
                            if beroep.lower() in title.lower():
                                is_person = False
                            beroep_list = []
                            if "<br />" in beroep:
                                beroep_list = beroep.split("<br />")
                            elif "<br>" in beroep:
                                beroep_list = beroep.split("<br>")
                            elif "<br/>" in beroep:
                                beroep_list = beroep.split("<br/>")
                            else:
                                beroep_list = beroep.split(",")
                            for i, b in enumerate(beroep_list):
                                beroep_list[i] = b.replace("[", "").replace("]", "").replace("<small>", "").replace(
                                    "</small>", "").strip()
                            occupations += beroep_list

                categories = re.findall(categories_regex, elem.text)
                if categories:
                    pattern = r'\[\[Categorie:(.+?)(\|.*)?\]\]'
                    categories_list = []
                    for category_str in categories:
                        match = re.match(pattern, category_str)
                        if match:
                            category = match.group(1)
                            categories_list.append(category)
                    for c in categories_list:
                        if not is_person:
                            if "persoon" in c.lower():
                                is_person = True

                    occupations_set = set(occupation_list)
                    categories_set = set(categories_list)
                    occupations_in_categories = occupations_set.intersection(categories_set)
                    if occupations_in_categories:
                        if title in list(occupations_in_categories):
                            is_person = False
                        else:
                            is_person = True
                        occupations += list(occupations_in_categories)

                    other_categories = categories_set - occupations_in_categories
                    other_categories = list(other_categories)

                    shortened_categories = []
                    relevant_indices = []
                    for i in range(len(other_categories)):
                        words = other_categories[i].split(" ")
                        if len(words) > 1:
                            shortened_categories.append(' '.join(words[1:]).capitalize())
                            relevant_indices.append(i)
                    shortened_categories_set = set(shortened_categories)
                    occupations_in_shortened_categories = occupations_set.intersection(shortened_categories_set)
                    if occupations_in_shortened_categories:
                        is_person = True
                        for j in relevant_indices:
                            occupations.append(other_categories[j])

                    occupations = list(set(occupations))
                summary, section_list = clean_text(elem.text)
                if not is_person:

                    continue

                else:
                    yield title, summary, section_list, occupations, other_categories
            elem.clear()


def main():
    filter_wikipedia_dump()


if __name__ == '__main__':
    main()