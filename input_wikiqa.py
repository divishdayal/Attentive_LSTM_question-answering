"""Input utilities for the WikiQA dataset."""

from __future__ import print_function

import csv
import nltk


def load_data(filename, test=False):
    """
    Load WikiQA TSV data into a list of question/answer threads.

    Returns a list of entries shaped like:
    [question_text, [[answer, label], ... positives], [[answer, label], ... negatives]]
    """
    with open(filename, encoding='utf-8') as tsvfile:
        current_q = ''
        last_q = ''
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        current_thread_pos = []
        current_thread_neg = []
        threads = []
        first = True
        for line in tsvreader:
            answer_text = line[5]
            answer_ascii = answer_text.encode('ascii', 'ignore').decode('ascii')
            if line[0] == 'QuestionID':
                continue
            current_q = line[0]
            question_text = line[1]
            if current_q == last_q or first:
                if line[6] == '1':
                    current_thread_pos.append([answer_ascii, line[6]])
                elif line[6] == '0':
                    current_thread_neg.append([answer_ascii, line[6]])
                else:
                    print('error in reading')
                if first:
                    first = False
            else:
                threads.append([question_text, current_thread_pos,
                                current_thread_neg])
                current_thread_pos = []
                current_thread_neg = []
                if line[6] == '1':
                    current_thread_pos.append([answer_ascii, line[6]])
                elif line[6] == '0':
                    current_thread_neg.append([answer_ascii, line[6]])
                else:
                    print('error in reading')
            last_q = current_q
        threads.append([question_text, current_thread_pos, current_thread_neg])

    pos_threads = [
        thread for thread in threads
        if len(thread[1]) > 0 and len(thread[2]) > 0
    ]
    pos_threads_single = [thread for thread in pos_threads if len(thread[1]) == 1]
    pos_threads_multi = [thread for thread in pos_threads if len(thread[1]) > 1]
    for thread in pos_threads_multi:
        for answer in thread[1]:
            pos_threads_single.append([thread[0], answer, thread[2]])

    # pos_threads is the threads with at least one positive post only
    print('number of filtered data points - ', len(pos_threads_single))
    print('number of total data points - ', len(threads))
    return pos_threads_single


if __name__ == '__main__':
    nltk.download('punkt', quiet=True)
    print(load_data('WikiQA-test.tsv')[0])
