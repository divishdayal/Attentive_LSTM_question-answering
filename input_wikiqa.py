import csv
from chardet import detect
import numpy as np

def load_data(filename, test = False):
	with open(filename) as tsvfile:
		current_q = ''
		last_q = ''
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		current_thread_pos = []
		current_thread_neg = []
		threads = []
		first = 1
		for line in tsvreader:
			l = unicode(line[5], encoding='utf-8')
			if line[0] == 'QuestionID':
				continue
			current_q = line[0]
			question_text = line[1]
			if current_q == last_q or first:
				if line[6] == '1':
					current_thread_pos.append([l.encode('ascii', 'ignore'), line[6]])
				elif line[6] == '0':
					current_thread_neg.append([l.encode('ascii', 'ignore'), line[6]])
				else:
					print 'error in reading'
				if line[6] == 1:
					atleast_one = 1
				if first == 1:
					first = 0
			else:
				threads.append([question_text, current_thread_pos, current_thread_neg])
				current_thread_pos = []
				current_thread_neg = []
				if line[6] == '1':
					current_thread_pos.append([l.encode('ascii', 'ignore'), line[6]])
				elif line[6] == '0':
					current_thread_neg.append([l.encode('ascii', 'ignore'), line[6]])
				else:
					print 'error in reading'
			last_q = current_q
		threads.append([question_text, current_thread_pos, current_thread_neg])

	pos_threads = filter(lambda x: len(x[1]) > 0 and len(x[2]) > 0, threads)
	pos_threads_single = filter(lambda x: len(x[1]) == 1, pos_threads)
	pos_threads_multi = filter(lambda x: len(x[1]) > 1, pos_threads)
	for x in pos_threads_multi:
		for i in x[1]:
			pos_threads_single.append([x[0], i, x[2]])

	#pos_threads is the theads with atleast one positive post only
	print 'number of filtered data points - ', len(pos_threads_single)
	print 'number of total data points - ', len(threads)
	return pos_threads_single
print load_data('WikiQA-test.tsv')[0]