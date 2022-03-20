import re
from typing import Generator
from os import path, listdir, getcwd


RAW_DATA_PATH = path.join(getcwd(), 'data', 'raw')
INTERIM_DATA_PATH = path.join(getcwd(), 'data', 'interim')


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
        dct[key] = m.group() if m is not None else ''

    # find abstract
    m = re.search('abstract = \"((?s:.+?))\"', s)
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


def process_abstract(s: str) -> str:
    if s == 'None':
        return s
    table = (
        ('\n', ' '),
    )
    for old, new in table:
        s = s.replace(old, new)
    return s


if __name__ == '__main__':
    from functools import reduce

    journals = ['ieee-tac', 'ieee-ted', 'ieee-proc']
    kw = []
    for journal in journals:
        data_dir = path.join(RAW_DATA_PATH, journal)
        keywords = set()
        for filename in listdir(data_dir):
            if not filename.endswith('.txt'):
                continue
            entries = read_db_file(path.join(data_dir, filename))
            for entry in entries:
                d = read_raw_entry(entry)
                d['abstract'] = process_abstract(d['abstract'])
                if 'keywords' in d and 'IEEE Keywords' in d['keywords']:
                    keywords.update(d['keywords']['IEEE Keywords'])
        kw.append(keywords)
    for j, keywords in zip(journals, kw):
        print(j, len(keywords))
    keywords = reduce(lambda a, b: set.union(a, b), kw)
    print(f'total: {len(keywords)}')
