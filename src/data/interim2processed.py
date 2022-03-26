import re
import os
from typing import Generator


INTERIM_DATA_PATH = os.path.join(os.getcwd(), 'data', 'interim')
PROCESSED_DATA_PATH = os.path.join(os.getcwd(), 'data', 'processed')


def main():
    journals = ('ieee-proc', 'ieee-tac', 'ieee-ted', 'ieee-tcom')
    NUM_KEYWORDS = 150
    OUTPUT_DB_FILE = 'dataset.csv'
    OUTPUT_KW_FILE = 'keywords.txt'

    # iterate over all papers and count keyword occurencies
    kw_dict = {}
    for journal in journals:
        fname = os.path.join(INTERIM_DATA_PATH, journal + '.txt')
        for dct in get_entries(fname):
            for key in dct['keywords']:
                kw_dict[key] = kw_dict.get(key, 0) + 1
    print(f'total: {len(kw_dict)} unique keywords')

    # sort keywords by frequency
    kw_list = list(kw_dict.items())
    kw_list.sort(key=lambda t: t[1], reverse=True)
    top_keywords = [kw_list[i][0] for i in range(NUM_KEYWORDS)]

    # write abstracts to database file
    db_fname = os.path.join(PROCESSED_DATA_PATH, OUTPUT_DB_FILE)
    with open(db_fname, 'w') as db_file:
        db_file = open(db_fname, 'w')
        db_file.write(','.join(('journal', 'DOI', 'abstract', 'keywords'))
                      + '\n')
        # write journal name  and DOI for identifying papers
        # keywords is a binary vector

        # iterate over all papers again and write abstracts to dataset file
        # if at least one of `NUM_KEYWORDS` most common keywords occurs
        for journal in journals:
            fname = os.path.join(INTERIM_DATA_PATH, journal + '.txt')
            for dct in get_entries(fname):
                kw_mask = ['0'] * NUM_KEYWORDS
                has_top_keyword = False
                for i, keyword in enumerate(top_keywords):
                    if keyword in dct['keywords']:
                        kw_mask[i] = '1'
                        has_top_keyword = True
                if has_top_keyword:
                    line = ','.join(
                        map(
                            lambda s: '"' + s + '"',
                            (journal, dct['DOI'], dct['abstract'],
                             ''.join(kw_mask))
                        )
                    )
                    db_file.write(line + '\n')

    # write keywords
    kw_fname = os.path.join(PROCESSED_DATA_PATH, OUTPUT_KW_FILE)
    with open(kw_fname, 'w') as kw_file:
        for keyword in top_keywords:
            kw_file.write(keyword + '\n')


def get_entries(fname: str) -> Generator[dict, None, None]:
    """
    Read text file `fname` with processed raw entries (no missing fields, no
    newlines in abstract) and return dictionaries with article data.
    """
    def match_field(field: str) -> str:
        # match "field = [value]" from next line in input file
        return re.match(f'{field} = (.*)\n$', next(infile)).groups()[0]

    with open(fname, 'r') as infile:
        while True:
            dct = {}
            try:
                line = next(infile)
                assert line == '[paper]\n'
            except StopIteration:
                return
            dct['title'] = match_field('title')
            dct['authors'] = match_field('authors').split(';')
            dct['journal'] = match_field('journal')
            dct['year'] = int(match_field('year'))
            dct['volume'] = match_field('volume')
            dct['issue'] = match_field('issue')
            dct['abstract'] = match_field('abstract')
            dct['paper citations'] = int(match_field('paper citations'))
            dct['patent citations'] = int(match_field('patent citations'))
            views = match_field('full text views')
            dct['full text views'] = int(views) if views != 'None' else None
            dct['open access'] = bool(match_field('open access'))
            dct['DOI'] = match_field('DOI')
            dct['URL'] = match_field('URL')
            dct['keywords'] = match_field('IEEE Keywords').split(';')
            next(infile)  # skip empty line
            yield dct


if __name__ == '__main__':
    main()
