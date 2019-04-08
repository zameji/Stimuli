import sys
import csv
import tqdm
import psutil
import random
import time
from numpy import array, digitize, zeros, amin, amax, arange, percentile, histogram, argmin, transpose, delete, log, exp
from numpy import random as nrandom
from numpy import sum as nsum
from math import floor, ceil
from nltk import word_tokenize, pos_tag
import matplotlib.pyplot as plt
from copy import deepcopy
from re import match, sub, compile
import json
import os
import gc
gc.enable()

def getPos(bigram):
	bigram = word_tokenize(bigram)
	bigram = pos_tag(bigram)
	# bigram = "_".join([x[1][0] for x in bigram])	
	return(bigram)
	
def getBins(it2score, cuts=20, type="dict", method = "exact", percent=95):
	"""Extract the range of scores. Take dict (or dict of dicts if specified), return lower/upper bound. Returns numpy array of edges."""
	if type == "dict":
		try:
			scores = [it2score[x] for x in it2score]							#	Get from dictionary to list
		except:
			print("Conversion not possible. Wrong type selected?")
	elif type == "dictOfDicts":
		scores = [it2score[x][y] for x in it2score for y in it2score[x]]	#	Get from dictionary to list
	
	else:
		print("Dicts nested more than once not implemented")
		raise ValueError()
		
	scores = array(scores)
	
	print("        Cropping extreme %i percent of data" % (100-percent))
	lower, upper = percentile(scores, [(100-percent)/2, ((100-percent)/2)+percent])
	scores = scores[(scores > lower) & (scores < upper)]
	
	if method == "log":
		print("         Defining log-transform-based bin boundaries")
		# bins = histogram_bin_edges(scores, cuts)
		
		translate = (0-amin(scores)) + 1
		#print(amin(scores))
		#print(amax(scores))
		scores = log(scores + translate)

		counts, bins = histogram(scores, cuts-2)
		# print(counts/sum(counts))
		bins = exp(bins) - translate
		scores = exp(scores) - translate
		
	else:
		print("         Defining bin boundaries")
		# bins = histogram_bin_edges(scores, cuts)
		
		#translate = (0-amin(scores)) + 1
		#print(amin(scores))
		#print(amax(scores))
		#scores = log(scores + translate)

		counts, bins = histogram(scores, cuts-2)
		# print(counts/sum(counts))
		#bins = exp(bins) - translate
		#scores = exp(scores) - translate
	#plt.hist(scores, bins) 
	#plt.show()	
	
	# x = input("Press RETURN to continue")
	#print(bins)
	return(bins)
	
def getDisbalance(new, old, penalty=3):
	old = deepcopy(old)
	for item in new:
		old[arange(old.shape[0]), item] += 1
	m = sum(old[0,:])/old.shape[1]
	old = old - m
	old == old**penalty
	return(nsum(old))

def getGain(new, old, penalty=3):
	old = deepcopy(old)
	for item in new:
		old[arange(old.shape[0]), item] -= 1
	m = sum(old[0,:])/old.shape[1]
	old = old - m
	old == old**penalty
	return(nsum(old))
	
class Scorer(object):
	"""Loads all the score lists, can then be used to asign the scores on a per-item base with the score method.
		It is memory-heavy, but could be included in functions which allow interactive collocation input."""
		
	def __init__(self):
		"""Load the required score files. If they are not present in the folder, throw an exception."""
		self.binning = "exact"
		
		import os		
		files = ["wfreqs.json", "bigrams.json","fwd.json","bckw.json","llscore.json","dicescore.json","tscore.json","zscore.json", "delta_p21.json", "delta_p12.json", "miscore.json","mi3score.json","gscore.json"]
		files = ["scores2/"+x for x in files]
		try:
			filecheck = [os.path.isfile(f) for f in files]
			if all(filecheck) != True:
				raise IOError()

		except:
			print("\nFollowing files could not be loaded. Check that they are in the /scores subfolder as this script.")
			for i in range(len(files)):
				if filecheck[i]==False:
					print(files[i])
			print("Exiting")
			sys.exit(1)		
		
		try:
			print("\nLoading the saved scores")
			print("    unigram frequency")
			with open("scores2/wfreqs.json", "r") as i:
				self.wfreq = json.loads(i.read())	
			
			print("    bigram frequency")
			with open("scores2/bigrams.json", "r") as i:
				self.bg_frq = json.loads(i.read())
								
			print("    TP-D")	
			with open("scores2/fwd.json", "r") as i:
				self.tp_d = json.loads(i.read())	

			print("    TP-B")	
			with open("scores2/bckw.json", "r") as i:
				self.tp_b = json.loads(i.read())	

			print("    Log likelihood")	
			with open("scores2/llscore.json", "r") as i:
				self.log_lklhd = json.loads(i.read())	

			print("    Modified dice")	
			with open("scores2/dicescore.json", "r") as i:
				self.dice = json.loads(i.read())	

			print("    t-score")	
			with open("scores2/tscore.json", "r") as i:
				self.t_score = json.loads(i.read())	

			print("    z-score")	
			with open("scores2/zscore.json", "r") as i:
				self.z_score = json.loads(i.read())

			print("    delta_p-12")
			with open("scores2/delta_p12.json", "r") as i:
				self.delta_p12 = json.loads(i.read())	
				
			print("    delta_p-21")
			with open("scores2/delta_p21.json", "r") as i:
				self.delta_p21 = json.loads(i.read())					

			print("    MI-score")	
			with open("scores2/miscore.json", "r") as i:
				self.mi_score = json.loads(i.read())	
				
			print("    MI3-score")	
			with open("scores2/mi3score.json", "r") as i:
				self.mi3_score = json.loads(i.read())	

			print("    G-score")	
			with open("scores2/gscore.json", "r") as i:
				self.g_score = json.loads(i.read())	
			print("_________________________________")
			
		except:
			print("Not all score files could be loaded. Check that they are in the same folder as this script.")
			print("Exiting")
			sys.exit(1)

	def score(self, items):
		"""Score all the bigrams at once. Input is a list of lists with the sublist format
		[item1, item2]. If a bigram is not in the score list, return NA."""
		
		items_out = []
		
		for bigram in items:
			w1, w2 = bigram
			bigram = " ".join(bigram)

			try:
				bg_frq = self.bg_frq[bigram]
			except:
				bg_frq = "NA"		

			try:
				tp_d = self.tp_d[w1][w2]
			except:
				tp_d = "NA"
				
			try:
				tp_b = self.tp_b[w2][w1]
			except:
				tp_b = "NA"
				
			try:
				log_lklhd = self.log_lklhd[bigram]
			except:
				log_lklhd = "NA"		
				
			try:
				dice = self.dice[bigram]
			except:
				dice = "NA"			
				
			try:
				t_score = self.t_score[bigram]
			except:
				t_score = "NA"			
				
			try:
				z_score = self.z_score[bigram]
			except:
				z_score = "NA"			
				
			try:
				mi_score = self.mi_score[bigram]
			except:
				mi_score = "NA"			

			try:
				mi3_score = self.mi3_score[bigram]
			except:
				mi3_score = "NA"	
				
			try:
				g_score = self.g_score[bigram]
			except:
				g_score = "NA"			

			try:
				delta_p12 = self.delta_p12[bigram]
			except:
				delta_p12 = "NA"	
				
			try:
				delta_p21 = self.delta_p21[bigram]
			except:
				delta_p21 = "NA"					
			
			items_out.append([bigram, bg_frq, tp_b, tp_d, log_lklhd, dice, t_score, z_score, mi_score, mi3_score, g_score, delta_p12, delta_p21])
	
		return(items_out)
		
		
	def score_one(self, bigram):
		"""Score the individual bigrams. Input is in the format
		[item1, item2]. If a bigram is not in the score list, return NA."""
	
		w1, w2 = bigram
		bigram = " ".join(bigram)

		try:
			bg_frq = self.bg_frq[bigram]
		except:
			bg_frq = "NA"		

		try:
			tp_d = self.tp_d[w1][w2]
		except:
			tp_d = "NA"
			
		try:
			tp_b = self.tp_b[w2][w1]
		except:
			tp_b = "NA"
			
		try:
			log_lklhd = self.log_lklhd[bigram]
		except:
			log_lklhd = "NA"		
			
		try:
			dice = self.dice[bigram]
		except:
			dice = "NA"			
			
		try:
			t_score = self.t_score[bigram]
		except:
			t_score = "NA"			
			
		try:
			z_score = self.z_score[bigram]
		except:
			z_score = "NA"			
			
		try:
			mi_score = self.mi_score[bigram]
		except:
			mi_score = "NA"			

		try:
			mi3_score = self.mi3_score[bigram]
		except:
			mi3_score = "NA"	
			
		try:
			g_score = self.g_score[bigram]
		except:
			g_score = "NA"			

		try:
			delta_p12 = self.delta_p12[bigram]
		except:
			delta_p12 = "NA"	
			
		try:
			delta_p21 = self.delta_p21[bigram]
		except:
			delta_p21 = "NA"	
			
		return([bigram, bg_frq, tp_b, tp_d, log_lklhd, dice, t_score, z_score, mi_score, mi3_score, g_score, delta_p12, delta_p21])		
		
	def get_random(self, num, cuts=20, seed=1991, disbalance_penalty = 3, max_time=20, words = ["", ""], pos=["",""],
		max_bg_freq=100000000, min_bg_freq=0, min_uni_freq=5, percent=95):
		"""Get num random bigrams, spread approximately evenly accross the ranges of the scores."""
		print("\nFinishing initialization")
		### FILTERING BY POS
		
		self.bgs_all = [x.split() for x in tqdm.tqdm(self.bg_frq) if (" ".join(x.split()) in self.bg_frq)]
		
		if (min_bg_freq	> 0) or (max_bg_freq < 100000000):
			print("    Selecting bigrams by bigram frequency")				
			self.bgs = [[x,y] for x,y in tqdm.tqdm(self.bgs_all) if (self.bg_frq[" ".join([x,y])] > min_bg_freq) and (self.bg_frq[" ".join([x,y])] < max_bg_freq)]	
		else:
			self.bgs = self.bgs_all
			
		if min_uni_freq > 1:
			print("    Selecting bigrams by unigram frequency")		
			self.bgs = [[x,y] for x,y in tqdm.tqdm(self.bgs) if (self.wfreq[x]>=min_uni_freq and self.wfreq[y]>=min_uni_freq)]	
		else:
			self.bgs = self.bgs
			
		if pos[0] != "" or pos[1] != "":	
			pos_1, pos_2 = pos
			pos_1 = compile(pos_1)
			pos_2 = compile(pos_2)
			print("    Selecting bigrams by POS")
			self.bgs = [[x,y] for x,y in tqdm.tqdm(self.bgs) if (match(pos_1, x.split("_")[1])!=None and match(pos_2,y.split("_")[1])!=None)]

		if words[0] != "" or words[1] != "":	
			w_1, w_2 = words
			w_1 = compile(w_1)
			w_2 = compile(w_2)
			print("    Selecting bigrams by words 1 & 2")
			self.bgs = [[x,y] for x,y in tqdm.tqdm(self.bgs) if (match(w_1, x.split("_")[0])!=None and match(w_2,y.split("_")[0])!=None)]
		
		
		print("    Cleaning unfitting bigrams")
		print("          Initializing cleaner")

		tempbgs = set([" ".join(x) for x in self.bgs])
		wrong_keys = [x for x in tqdm.tqdm(self.bg_frq) if not x in tempbgs]
		del tempbgs
		print("          Cleaning")
		wrong_keys = []
		for wrong_key in tqdm.tqdm(wrong_keys):
			try:
				c = wrong_key
				wrong_key = wrong_key.split()
				del self.bg_frq[c]
				del self.tp_d[wrong_key[0]][wrong_key[1]]
				del self.tp_b[wrong_key[1]][wrong_key[0]]
				del self.log_lklhd[c]
				del self.dice[c]
				del self.t_score[c]
				del self.z_score[c]
				del self.mi_score[c]
				del self.mi3_score[c]
				del self.g_score[c]
				del self.delta_p12[c]
				del self.delta_p21[c]
			except:
				pass
			
		del wrong_keys
		
		print("\nDefining data distributions")
		print("    bigram frequency")
		self.bg_frq_bins = getBins(self.bg_frq, cuts=cuts, method = self.binning, percent=percent)						#	Get from dictionary to list
		
		print("    TP-D")	
		
		self.tp_d_bins = getBins(self.tp_d, cuts=cuts, type="dictOfDicts", percent=percent)		#	Get from dictionary to list

		print("    TP-B")	
		self.tp_b_bins = getBins(self.tp_b, cuts=cuts, type="dictOfDicts", percent=percent)		#	Get from dictionary to list

		print("    Log likelihood")	
		self.log_lklhd_bins = getBins(self.log_lklhd, cuts=cuts, percent=percent)				#	Get from dictionary to list

		print("    Modified dice")	
		self.dice_bins = getBins(self.dice, cuts=cuts, percent=percent)							#	Get from dictionary to list

		print("    t-score")	
		self.t_score_bins = getBins(self.t_score, cuts=cuts, percent=percent)					#	Get from dictionary to list

		print("    z-score")	
		self.z_score_bins = getBins(self.z_score, cuts=cuts, percent=percent)					#	Get from dictionary to list
		
		print("    MI-score")	
		self.mi_score_bins = getBins(self.mi_score, cuts=cuts, percent=percent)					#	Get from dictionary to list
			
		print("    MI3-score")	
		self.mi3_score_bins = getBins(self.mi3_score, cuts=cuts, percent=percent)				#	Get from dictionary to list

		print("    G-score")	
		self.g_score_bins = getBins(self.g_score, cuts=cuts, percent=percent)					#	Get from dictionary to list
		
		print("    Delta_p-12")	
		self.delta_p12_bins = getBins(self.delta_p12, cuts=cuts, percent=percent)					#	Get from dictionary to list

		print("    Delta_p-21")	
		self.delta_p21_bins = getBins(self.delta_p21, cuts=cuts, percent=percent)					#	Get from dictionary to list		
		
		print("_________________________________")
		
		
		buf = 10																# The bufferring coefficient (how many extra elements should be collected)
		self.dist = zeros([12, cuts])											# 	Array to save the distributions: columns=bins, rows=scores
		self.results = zeros([num*buf, 12])										#	Array to save the results
		self.items = []
		self.populated = 0
		self.disbalance_penalty = floor(num*disbalance_penalty/100)
		# self.indexes = arange(length(self.bg_pos))
		
		random.seed(seed)		
		nrandom.seed(seed)
		
		print("\nStarting item selection")

		start = time.time()
		lasttime = time.time()			
		
		max_time = max_time*60
		pbar = [tqdm.tqdm(total = num), tqdm.tqdm(total=num*(buf-1))]
		pbar[0].set_description("Stimuli collected")
		pbar[1].set_description("Additional buffer")
		reached = False
		
		while (self.populated < num*buf) and (time.time() - start) < max_time:				#	Get a random item; check which bins would it increase for which score, if this disturbs balance, drop otherwise insert at the bottom
			if self.populated >= num and reached == False:
				reached = True
				
			samples = [nrandom.random_integers(0,len(self.bgs)-1,10) for x in range(1000)]
			samples = [[self.bgs[y] for y in x] for x in samples]
			scores = [self.score(x) for x in samples]										# remove bigram string to allow numpy operation
			scores = [array([x[1:] for x in y]) for y in scores]
			
			binned = [array([digitize(x[:,0],self.bg_frq_bins),
						digitize(x[:,1],self.tp_b_bins),						
						digitize(x[:,2],self.tp_d_bins),
						digitize(x[:,3],self.log_lklhd_bins),
						digitize(x[:,4],self.dice_bins),
						digitize(x[:,5],self.t_score_bins),
						digitize(x[:,6],self.z_score_bins),
						digitize(x[:,7],self.mi_score_bins),
						digitize(x[:,8],self.mi3_score_bins),
						digitize(x[:,9],self.g_score_bins),
						digitize(x[:,10],self.delta_p12_bins),
						digitize(x[:,11],self.delta_p21_bins)]) for x in scores]	
			
			# print(binned[0])
			binned = [transpose(x) for x in binned]
			# binned = []
			performance = [getDisbalance(x, self.dist, self.disbalance_penalty) for x in binned]
			best = argmin(performance)

			self.results[self.populated:self.populated+10,:] = scores[best]
			for item in binned[best]:
				self.dist[arange(12), item] += 1
			self.items += samples[best]
			self.populated += 10
			if reached:
				pbar[1].update(10)
			else:
				pbar[0].update(10)
		
		pbar[0].close()
		pbar[1].close()
		
		if self.populated >= num and (time.time() - start) < max_time:
			print("\nPruning")
				
			binned = array([digitize(self.results[:,0],self.bg_frq_bins),
						digitize(self.results[:,1],self.tp_b_bins),						
						digitize(self.results[:,2],self.tp_d_bins),
						digitize(self.results[:,3],self.log_lklhd_bins),
						digitize(self.results[:,4],self.dice_bins),
						digitize(self.results[:,5],self.t_score_bins),
						digitize(self.results[:,6],self.z_score_bins),
						digitize(self.results[:,7],self.mi_score_bins),
						digitize(self.results[:,8],self.mi3_score_bins),
						digitize(self.results[:,9],self.g_score_bins),
						digitize(self.results[:,10],self.delta_p12_bins),
						digitize(self.results[:,11],self.delta_p21_bins)])	
			binned = transpose(binned)

			print("Removing duplicates")
			firsts = set()
			seconds = set()
			dels = []
			for x in tqdm.tqdm([y for y in range(self.populated)]):
				w1,w2 = self.items[x]
				if w1 in firsts or w2 in seconds:
					dels.append(x)
				else:
					firsts.update(w1)
					seconds.update(w2)
			
			dels.reverse()
			for d_index in tqdm.tqdm(dels):
				del self.items[d_index]
			binned = delete(binned, dels,0)
			self.results = delete(self.results, dels, 0)
			self.populated -= len(dels)
			
			pbar = tqdm.tqdm(total=self.populated - num)
			while self.populated > num and (time.time() - start) < max_time:			# If there is time left, prune the most problematic items away, one by one
				rands = [random.randint(0,binned.shape[0]-1) for x in range(1000)]
				performance = [getGain(binned[i,:], self.dist, self.disbalance_penalty) for i in rands]
				best = rands[argmin(performance)]
				self.dist[arange(12), binned[best,:]] -= 1
				binned = delete(binned, best, 0)
				del self.items[best]
				self.results = delete(self.results, best, 0)
				self.populated -= 1
				pbar.update(1)
				
			print("\nSuccess! All %i items were found." % num)
			
			
		else:
			print("\nTimeout limit exceeded. Returning %i items" % self.populated)
			results = self.results[0:min(self.populated, num)]							# Crop which we don't have

		#plt.imshow(self.dist, cmap="hot", interpolation="bilinear")
		#plt.suptitle("Distribution accross scores")
		#plt.xlabel("Score bin")
		#plt.ylabel("Score")
		#plt.show()
		
		results = [[sub("_[^ ]+",""," ".join(self.items[x]))]+list(self.results[x, 0:12]) for x in range(min(self.populated, num))]
		
		return(results)

class ScorerLite(object):
	"""This class is used for extracting the scores on low-RAM machines (<8GB seems like a good cutoff).
	The individul scores are loaded on per-demand base and the scores are asigned one score at a time:
	This requires all items to be chosen prior to the asignment, otherwise the program becomes inefficient.
	"""
	
	def __init__(self):
		"""Checks that the needed files are there, does not load them."""
		
		import os		
		files = ["wfreqs.json", "bigrams.json","fwd.json","bckw.json","llscore.json","dicescore.json","tscore.json","zscore.json","miscore.json","mi3score.json", "delta_p12.json","delta_p21.json","gscore.json"]
		files = ["scores2/"+x for x in files]
		try:
			filecheck = [os.path.isfile(f) for f in files]
			if all(filecheck) != True:
				raise IOError()

		except:
			print("Following files could not be loaded. Check that they are in the /scores subfolder as this script.")
			for i in range(len(files)):
				if filecheck[i]==False:
					print(files[i])
			print("Exiting")
			sys.exit(1)		

	def score(self, items):
		"""Assigns all the scores to all the items. 
		Input is a list of lists, where each sublist contains 2 elements: [word1, word2]"""
			
		items_out = [[" ".join(x)] for x in items]
		bigrams = [" ".join(x) for x in items]
		
		print("Loading the saved scores and asigning them")
		print("    bigram frequency")
		with open("scores2/bigrams.json", "r") as i:
			bg_frq = json.load(i)
			for i in range(len(items)):
				try:
					items_out[i].append(bg_frq[bigrams[i]])
				except:
					items_out[i].append("NA")
			del bg_frq				
									
			
									
		print("    TP-B")	
		with open("scores2/bckw.json", "r") as i:
			tp_b = json.loads(i.read())
			for i in range(len(items)):
				try:
					items_out[i].append(tp_b[items[i][1]][items[i][0]])
				except:
					items_out[i].append("NA")
			del tp_b	
			
		print("    TP-D")	
		with open("scores2/fwd.json", "r") as i:
			tp_d = json.loads(i.read())
			for i in range(len(items)):
				try:
					items_out[i].append(tp_d[items[i][0]][items[i][1]])
				except:
					items_out[i].append("NA")
			del tp_d	
			
		print("    Log likelihood")	
		with open("scores2/llscore.json", "r") as i:
			log_lklhd = json.loads(i.read())	
			for i in range(len(items)):
				try:
					items_out[i].append(log_lklhd[bigrams[i]])
				except:
					items_out[i].append("NA")
			del log_lklhd				
							
		print("    Modified dice")	
		with open("scores2/dicescore.json", "r") as i:
			dice = json.loads(i.read())
			for i in range(len(items)):
				try:
					items_out[i].append(dice[bigrams[i]])
				except:
					items_out[i].append("NA")
			del dice				
							
		print("    t-score")	
		with open("scores2/tscore.json", "r") as i:
			t_score = json.loads(i.read())	
			for i in range(len(items)):
				try:
					items_out[i].append(t_score[bigrams[i]])
				except:
					items_out[i].append("NA")
			del t_score				
							
		print("    z-score")	
		with open("scores2/zscore.json", "r") as i:
			z_score = json.loads(i.read())	
			for i in range(len(items)):
				try:
					items_out[i].append(z_score[bigrams[i]])
				except:
					items_out[i].append("NA")
			del z_score				
							
		print("    MI-score")	
		with open("scores2/miscore.json", "r") as i:
			mi_score = json.loads(i.read())	
			for i in range(len(items)):
				try:
					items_out[i].append(mi_score[bigrams[i]])
				except:
					items_out[i].append("NA")
			del mi_score				
											
		print("    MI3-score")	
		with open("scores2/mi3score.json", "r") as i:
			mi3_score = json.loads(i.read())	
			for i in range(len(items)):
				try:
					items_out[i].append(mi3_score[bigrams[i]])
				except:
					items_out[i].append("NA")
			del mi3_score				
							
		print("    G-score")	
		with open("scores2/gscore.json", "r") as i:
			g_score = json.loads(i.read())	
			for i in range(len(items)):
				try:
					items_out[i].append(g_score[bigrams[i]])
				except:
					items_out[i].append("NA")
			del g_score				

		print("    Delta_p-12-score")	
		with open("scores2/delta_p12.json", "r") as i:
			delta_p12 = json.loads(i.read())	
			for i in range(len(items)):
				try:
					items_out[i].append(delta_p12[bigrams[i]])
				except:
					items_out[i].append("NA")
			del delta_p12	
			
		print("    Delta_p-21-score")	
		with open("scores2/delta_p21.json", "r") as i:
			delta_p21 = json.loads(i.read())	
			for i in range(len(items)):
				try:
					items_out[i].append(delta_p21[bigrams[i]])
				except:
					items_out[i].append("NA")
			del delta_p21				
		
		return(items_out)
						
if __name__ == "__main__":
	mode = None
	while mode not in ["score", "search", "strat_search"]:
		mode = input("Which mode should this program run in?\n Score/search: ")
	
	if mode.lower() == "score":
		
		if len(sys.argv) > 1:
			inpath = sys.argv[1]
		else:
			inpath = raw_input("Where is the file to load?\n     ")
			
		try:
			with open(inpath, "r") as infile:
				items = infile.readlines()
				items = [x.split() for x in items]
				if all([len(x)==2 for x in items]) == False:
					raise IOError()

		except:
			print("The input file does not seem to be formatted correctly (one bigram per line)")
			sys.exit(2)

		ram_present = psutil.virtual_memory()[0] >> 30
		ram_available = psutil.virtual_memory()[1] >> 30

		# Check the RAM installed and available, if sufficient use the default scorer, otherwise use the lite version
		if ram_present > 7 and ram_available > 5:
			scorer = Scorer()	
		else:
			scorer = ScorerLite()
			
		items = scorer.score(items)	
			
		if len(sys.argv) > 2:
			outpath = sys.argv[2]
		else:
			outpath = raw_input("Where should the results be saved?\n    ")
		
		print("Saving")
		with open(outpath, "w+") as outfile:
			out_csv = csv.writer(outfile)
			out_csv.writerow(["bigram", "bigram_freq", "tp_b", "tp_d", "log_lklhd", "dice", "t_score", "z_score", "mi_score", "mi3_score", "g_score", "delta_p12", "delta_p21"])
			for i in tqdm.tqdm(items):
				out_csv.writerow(i)
		print("Done. Press RETURN to exit")
		wait = raw_input()
		sys.exit(0)

	elif mode.lower() == "search":
		
		ram_present = psutil.virtual_memory()[0] >> 30
		ram_available = psutil.virtual_memory()[1] >> 30

		# Check the RAM installed and available, if sufficient use the default scorer, otherwise use the lite version
		if ram_present > 7:
		# if ram_present > 7 and ram_available > 5:
			pass	
		else:
			print("WARNING: This is RAM-intensive operation. It cannot continue if you don't have at least 8 GB of RAM.\nExiting...")
			sys.exit(0)
			
		max_time = 0
		disbalance_penalty = 0
		cuts = 0
		num = 0
		pos = ""
		
		saved = False
		if os.path.isfile("search_settings.py"):
			print("Saved settings found. Do you want to use them? [Y/n]")
			dec = ""
			while dec not in set(["y", "n"]):
				dec = input("[Y/n]").lower()
				if dec == "":
					dec = "y"
				
			if dec == "y":			
				saved = True
				
		if saved == True:	
			from search_settings import *
			print("Using saved settings")
			
		else:
			
			while num <=0:
				try:
					num = int(input("How many bigrams do you want to find? "))
				except:
					num = 100
			while cuts <=0:
				try:
					cuts = int(input("\nHow many bins should the scores be divided into? "))
				except:
					cuts = 20
			while disbalance_penalty <=0:
				try:
					disbalance_penalty = float(input("\nHow large a penalty should be given to imbalance (1-5)? "))
				except: 
					disbalance_penalty = 2
			while max_time <=0:
				try:
					max_time = int(input("\nWhat is the time limit for the search (in minutes)?"))
				except:
					max_time = 5
			print("\nDo you want to filter the bigrams by their POS-tags?\nWrite the RegEx to match (the tags are lower-cased): E.g. j.* to extract adjectives\nLeave empty to ignore\n")
			pos_1 = input("\nWord 1: ")
			pos_2 = input("\nWord 2: ")
			
			print("\nDo you want to filter the bigrams by the string?\nWrite the RegEx to match: E.g. [a-z].* to extract lower-case only\nLeave empty to ignore\n")
			w_1 = input("\nWord 1: ")
			w_2 = input("\nWord 2: ")
			
			try:
				min_uni_freq = int(input("\nWhat should be the lowest unigram frequency included? "))
			except:
				min_uni_freq = 0
			try:
				min_bg_freq = int(input("\nWhat should be the lowest bigram frequency included? "))
			except:
				min_bg_freq = 0
			try:
				max_bg_freq = int(input("\nWhat should be the highest bigram frequency included? "))
			except:
				max_bg_freq = 0
				
			seed = 1991
		
		scorer = Scorer()		
		items = scorer.get_random(num, cuts=cuts, seed=seed, disbalance_penalty = disbalance_penalty, words=[w_1, w_2], pos=[pos_1, pos_2], max_time=max_time, min_bg_freq=min_bg_freq, max_bg_freq=max_bg_freq, min_uni_freq=min_uni_freq)
		
		print("Saving")
		
		if len(sys.argv) > 2:
			outpath = sys.argv[2]
		else:
			outpath = input("Where should the results be saved?\n    ")
		
		print("Saving")
		with open(outpath, "w+") as outfile:
			out_csv = csv.writer(outfile)
			out_csv.writerow(["bigram", "bigram_freq", "tp_b", "tp_d", "log_lklhd", "dice", "t_score", "z_score", "mi_score", "mi3_score", "g_score", "delta_p12", "delta_p21"])
			for i in tqdm.tqdm(items):
				out_csv.writerow(i)
				
		print("Do you want to save your settings? [y/N]")
		dec = ""
		while dec not in set(["y", "n"]):
			dec = input("[y/N]").lower()
			if dec == "":
				dec = "n"
			
		if dec == "y":
			settings = ""
			settings += "num = %i\n" % num
			settings += "cuts = %i\n" % cuts
			settings += "disbalance_penalty = %i\n" % disbalance_penalty
			settings += 'w_1 ="' + w_1 + '"\n'
			settings += 'w_2 ="' + w_2 + '"\n'
			settings += 'pos_1 ="' + pos_1 + '"\n'
			settings += 'pos_2 ="' + pos_2 + '"\n'
			settings += "max_time = %i\n" % max_time
			settings += "min_bg_freq = %i\n" % min_bg_freq
			settings += "max_bg_freq = %i\n" % max_bg_freq
			settings += "min_uni_freq = %i\n" % min_uni_freq

			with open("search_settings.py", "w+") as f:
				f.write(settings)

	elif mode.lower() == "strat_search":

		if len(sys.argv) > 2:
			outpath = sys.argv[2]
		else:
			outpath = input("Where should the results be saved?\n    ")

		with open(outpath, "w+") as outfile:
			out_csv = csv.writer(outfile)				
			out_csv.writerow(["bigram", "bigram_freq", "tp_b", "tp_d", "log_lklhd", "dice", "t_score", "z_score", "mi_score", "mi3_score", "g_score", "delta_p12", "delta_p21"])
		
		ram_present = psutil.virtual_memory()[0] >> 30
		ram_available = psutil.virtual_memory()[1] >> 30

		# Check the RAM installed and available, if sufficient use the default scorer, otherwise use the lite version
		if ram_present > 7:
		# if ram_present > 7 and ram_available > 5:
			pass	
		else:
			print("WARNING: This is RAM-intensive operation. It cannot continue if you don't have at least 8 GB of RAM.\nExiting...")
			sys.exit(0)
			
		max_time = 0
		disbalance_penalty = 0
		cuts = 0
		num = 0
		pos = ""
		
		saved = False
		if os.path.isfile("search_settings.py"):
			print("Saved settings found. Do you want to use them? [Y/n]")
			dec = ""
			while dec not in set(["y", "n"]):
				dec = input("[Y/n]").lower()
				if dec == "":
					dec = "y"
				
			if dec == "y":			
				saved = True
				
		if saved == True:	
			from search_settings import *
			print("Using saved settings")
			
		else:
			
			while num <=0:
				try:
					num = int(input("How many bigrams do you want to find? "))
				except:
					num = 100
			while cuts <=0:
				try:
					cuts = int(input("\nHow many bins should the scores be divided into? "))
				except:
					cuts = 20
			while disbalance_penalty <=0:
				try:
					disbalance_penalty = float(input("\nHow large a penalty should be given to imbalance (1-5)? "))
				except: 
					disbalance_penalty = 2
			while max_time <=0:
				try:
					max_time = int(input("\nWhat is the time limit for the search (in minutes)?"))
				except:
					max_time = 5
			print("\nDo you want to filter the bigrams by their POS-tags?\nWrite the RegEx to match (the tags are lower-cased): E.g. j.* to extract adjectives\nLeave empty to ignore\n")
			pos_1 = input("\nWord 1: ")
			pos_2 = input("\nWord 2: ")
			
			print("\nDo you want to filter the bigrams by the string?\nWrite the RegEx to match: E.g. [a-z].* to extract lower-case only\nLeave empty to ignore\n")
			w_1 = input("\nWord 1: ")
			w_2 = input("\nWord 2: ")
			
			try:
				min_uni_freq = int(input("\nWhat should be the lowest unigram frequency included? "))
			except:
				min_uni_freq = 0
			try:
				min_bg_freq = int(input("\nWhat should be the lowest bigram frequency included? "))
			except:
				min_bg_freq = 0
			try:
				max_bg_freq = int(input("\nWhat should be the highest bigram frequency included? "))
			except:
				max_bg_freq = 0
				
			seed = 1991
		
		scorer = Scorer()
		
		scorer.binning = "log"
		items = scorer.get_random(1, cuts=10, seed=seed, disbalance_penalty = disbalance_penalty, words=[w_1, w_2],
			pos=[pos_1, pos_2], max_time=max_time, min_bg_freq=min_bg_freq, max_bg_freq=max_bg_freq, min_uni_freq=min_uni_freq, percent=95)
		
		bins = [min_bg_freq] + list(scorer.bg_frq_bins) + [max_bg_freq]
		
		scorer.binning = "exact"
		print(bins)
		for strat in tqdm.tqdm(range(len(bins)-1)):
			items = scorer.get_random(floor(num/len(bins)), cuts=cuts, seed=seed, disbalance_penalty = disbalance_penalty, words=[w_1, w_2],
				pos=[pos_1, pos_2], max_time=max_time, min_bg_freq=floor(bins[strat]), max_bg_freq=ceil(bins[strat+1])+1, min_uni_freq=min_uni_freq, percent=100)			
		
			print("Saving")
			with open(outpath, "a") as outfile:
				out_csv = csv.writer(outfile)				
				for i in tqdm.tqdm(items):
					out_csv.writerow(i)
				
		print("Do you want to save your settings? [y/N]")
		dec = ""
		while dec not in set(["y", "n"]):
			dec = input("[y/N]").lower()
			if dec == "":
				dec = "n"
			
		if dec == "y":
			settings = ""
			settings += "num = %i\n" % num
			settings += "cuts = %i\n" % cuts
			settings += "disbalance_penalty = %i\n" % disbalance_penalty
			settings += 'w_1 ="' + w_1 + '"\n'
			settings += 'w_2 ="' + w_2 + '"\n'
			settings += 'pos_1 ="' + pos_1 + '"\n'
			settings += 'pos_2 ="' + pos_2 + '"\n'
			settings += "max_time = %i\n" % max_time
			settings += "min_bg_freq = %i\n" % min_bg_freq
			settings += "max_bg_freq = %i\n" % max_bg_freq
			settings += "min_uni_freq = %i\n" % min_uni_freq

			with open("search_settings.py", "w+") as f:
				f.write(settings)
				
		print("Done. Press RETURN to exit")
		wait = input()
		sys.exit(0)
