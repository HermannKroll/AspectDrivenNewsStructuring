from nnsummary.config import MULTI_DOCUMENT_SNIPPET_SIZE, NEW_ARTICLE_PERSON_INFORMATION


def generate_snippet_excerpt(text, person_entry):
    person_name_parts = person_entry[5]
    window = int(MULTI_DOCUMENT_SNIPPET_SIZE / 2)

    text_lower = text.lower()
    position = None
    for part in person_name_parts:
        if part in text_lower:
            position = text_lower.find(part)

    if position == None:
        raise ValueError(f'Cannot find {person_name_parts} in {text_lower}')

    if len(text) <= MULTI_DOCUMENT_SNIPPET_SIZE:
        return text

    selected_text = text
    left_c, right_c = None, None
    if position - window < 0:
        if len(text) > MULTI_DOCUMENT_SNIPPET_SIZE:
            # take the beginning characters
            selected_text = text[0:MULTI_DOCUMENT_SNIPPET_SIZE]

            left_c = ' '
            right_c = text[min(MULTI_DOCUMENT_SNIPPET_SIZE + 1, len(text) - 1)]

    else:
        if position + window < len(text):
            # take between pos - w and pos + w
            selected_text = text[max(0, position - window): min(position + window, len(text) - 1)]
            left_c = text[max(0, position - (window + 1))]
            right_c = text[min(position + window + 1, len(text) - 1)]

        else:
            # take the last x characters
            selected_text = text[len(text) - MULTI_DOCUMENT_SNIPPET_SIZE:]

            left_c = selected_text[min(0, len(text) - (MULTI_DOCUMENT_SNIPPET_SIZE + 1))]
            right_c = ' '

    # Remove cropped on right side
    if right_c.isalpha():
        selected_text = ' '.join([s for s in selected_text.split(' ')[:-1]])

    # Remove cropped on left side
    if left_c.isalpha():
        selected_text = ' '.join([s for s in selected_text.split(' ')[1:]])

    if len(selected_text.strip()) == 0:
        raise ValueError(f'Something went wrong during text snippet excerpt generation (text is empty: {text})')

    if len(selected_text) > MULTI_DOCUMENT_SNIPPET_SIZE:
        raise ValueError(
            f'Something went wrong during text snipped excerpt generation (text is longer than 160 chars: {text})')

    return selected_text


# Do something with spaces. Do not crop words
# put "Fragment" in the text
def main() -> int:
    texts_to_test = [
        """Lorem Churchill ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore 
        et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. 
        Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, 
        consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, 
        sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, 
        no sea takimata sanctus est Lorem ipsum dolor sit amet.""",

        """Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore 
        et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. 
        Stet clita kasd gubergren, no sea takimata sanctus Churchill est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, 
        consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, 
        sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, 
        no sea takimata sanctus est Lorem ipsum dolor sit amet.""",

        """Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore 
        et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. 
        Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, 
        consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, 
        sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, 
        Churchill no sea takimata sanctus est Lorem ipsum dolor sit amet.""",

        """Lorem ipsum dolor Churchill si.""",

        """LoremipsumdolorChurchillsi."""
    ]
    for t in texts_to_test:
        t = t.replace('\n', '')
        t_selc = generate_snippet_excerpt(t, NEW_ARTICLE_PERSON_INFORMATION[0])
        print(f'{len(t_selc)}: ""{t_selc}""')
    return 0


if __name__ == '__main__':
    main()
