import csv

with open("WikiQA-train.tsv") as tsvfile:
	current_q = ''
	last_q = ''
	tsvreader = csv.reader(tsvfile, delimiter="\t")
	current_thread = []
	threads = []
	first = 1
	for line in tsvreader:
		if line[0] == 'QuestionID':
			continue
		current_q = line[0]
		question_text = line[1]
		if current_q == last_q or first:
			current_thread.append([line[5], line[6]])
			if line[6] == 1:
				atleast_one = 1
			if first == 1:
				first = 0
		else:
			threads.append([question_text, current_thread])
			current_thread = [[line[5], line[6]]]
		last_q = current_q
threads.append([question_text, current_thread])

pos_threads = []
for thread in threads:
	flag = 0
	for x in thread[1]:
		if x[1] == '1':
			flag = 1
	if flag == 1:
		pos_threads.append(thread)

#pos_threads is the theads with atleast one positive post only
print len(pos_threads)

print len(threads)
 		
