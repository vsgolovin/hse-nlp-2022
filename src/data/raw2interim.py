import re
from typing import Generator, NoReturn, TextIO
import os


RAW_DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw')
INTERIM_DATA_PATH = os.path.join(os.getcwd(), 'data', 'interim')


def main():
    # every directory is a journal name
    for journal in os.listdir(RAW_DATA_PATH):
        input_path = os.path.join(RAW_DATA_PATH, journal)
        if not os.path.isdir(input_path):
            continue

        # process each journal separately and save results to a single file
        new_file = os.path.join(INTERIM_DATA_PATH, journal + '.txt')
        with open(new_file, 'w') as outfile:
            for input_file in os.listdir(input_path):
                input_file_full_path = os.path.join(input_path, input_file)
                for entry in read_db_file(input_file_full_path):
                    dct = read_raw_entry(entry)  # paper data
                    # check of both abstract and IEEE keywords are present
                    if (
                        dct['abstract'] == 'None'
                        or 'keywords' not in dct
                        or 'IEEE Keywords' not in dct['keywords']
                    ):
                        continue  # skip this paper
                    # replace newlines with spaces
                    dct['abstract'] = str.replace(dct['abstract'], '\n', ' ')
                    dct['abstract'] = str.replace(dct['abstract'], '"', '""')
                    # write updated entry to file
                    write_interim_entry(outfile, dct)


def read_db_file(filename: str) -> Generator[str, None, None]:
    """
    Read text database file `filename` and return all entries as strings.
    """
    with open(filename, 'r') as f:
        cur_entry = []
        for line in f:
            if line == '[paper]\n':
                if cur_entry:
                    while cur_entry[-1] == '\n':
                        cur_entry.pop()
                    yield ''.join(cur_entry)
                cur_entry = [line]
            elif len(cur_entry) > 0:
                cur_entry.append(line)
        if cur_entry:
            yield ''.join(cur_entry)


def read_raw_entry(s: str) -> dict:
    """
    Make a dictionary from raw database entry string. Handles newlines in
    abstract and missing keywords.
    """
    lines = s.split('\n')
    assert lines[0] == '[paper]'

    # get fields before the abstract ('key = value\n')
    dct = {}
    keys = ('title', 'authors', 'journal', 'year', 'volume', 'issue')
    for i, key in enumerate(keys):
        m = re.match(key + ' = (.+)', lines[i + 1])
        dct[key] = m.groups()[0] if m is not None else ''

    # find abstract
    m = re.search('abstract = \"((?s:.+?))\"\n', s)
    assert m and len(m.groups()) == 1
    dct['abstract'] = m[1]

    # fields after abstract and before keywords
    i = 8
    while not lines[i].startswith('paper citations'):
        i += 1
    keys = ('paper citations', 'patent citations', 'full text views',
            'open access', 'DOI', 'URL')
    for j, key in enumerate(keys):
        k, v = lines[i + j].split(' = ')
        assert k == key
        dct[key] = v

    # read keywords
    try:
        ind = lines.index('[paper.keywords]')
    except ValueError:
        return dct
    dct['keywords'] = {}
    keyword_text = '\n'.join(lines[ind:])
    for m in re.findall('(.+?) = \"((?s:.+?))\"', keyword_text):
        kw_key = m[0]
        kw_values = m[1].split(';')
        dct['keywords'][kw_key] = kw_values

    return dct


def write_interim_entry(outfile: TextIO, paper_entry: dict) -> NoReturn:
    outfile.write('[paper]\n')
    keys = ('title', 'authors', 'journal', 'year', 'volume', 'issue',
            'abstract', 'paper citations', 'patent citations',
            'full text views', 'open access', 'DOI', 'URL')
    for key in keys:
        outfile.write(f'{key} = {paper_entry[key]}\n')
    keywords_str = ';'.join(paper_entry['keywords']['IEEE Keywords'])
    outfile.write(f'IEEE Keywords = {keywords_str}\n\n')


if __name__ == '__main__':
    main()
