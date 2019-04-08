num = 2000							# number of stimuli
cuts = 20							# number of bins used to determine the distribution
disbalance_penalty = 3				# penalty for disbalanced distributions (disbalace equals sum[(real-uniform)**2])
w_1 ="^[a-z]+$"						# RegEx describing w1
w_2 ="^[a-z]+$"						# RegEx describing w2
pos_1 ="jj$"						# RegEx describing w1's POS tag
pos_2 ="n[^p].*"					# RegEx describing w2's POS tag
max_time = 240						# max minutes to spend
min_bg_freq = 5						# lowest accepted bigram frequency (<5 has no effect as only bigrams with freq >= 5  were kept when collecting the data)
max_bg_freq = 10000000
min_uni_freq = 5					# lowest accepted unigram frequency
seed = 1991							# ensures reproducible results