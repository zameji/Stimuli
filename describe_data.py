import pandas as pd
from numpy import amin, amax, histogram, percentile, transpose
import tqdm
from re import compile, match, sub

""""
This script describes the max/min distribution of data that is obtained by specific search settings. 
Eventually, it should become an element of the main script.
"""

class Data(object):
	def __init__(self):
		self.data = pd.read_csv("data/compact_scores.csv")
		
		print("Cleaning...")
		pbar = tqdm.tqdm(total=4)
		self.data = self.data[self.data["bigram"].str.contains("^\S+ \S+$")]
		pbar.update(1)
		self.data["w_1"], self.data["w_2"] = self.data["bigram"].str.split(" ", 1).str
		pbar.update(1)
		self.data["w_1"], self.data["pos_1"] = self.data["w_1"].str.split("_", 1).str
		pbar.update(1)
		self.data["w_2"], self.data["pos_2"] = self.data["w_2"].str.split("_", 1).str
		pbar.close()


	def describe(self, num, cuts=20, seed=1991, disbalance_penalty = 3, max_time=20, words = ["", ""], pos=["",""],
		max_bg_freq=100000000, min_bg_freq=0, min_uni_freq=5, percent=95):
		
		
		if (min_bg_freq	> 0) or (max_bg_freq < 100000000):
			print("    Selecting bigrams by bigram frequency")
			self.bgs = self.data[(self.data["bigram_freq"] >= min_bg_freq) & (self.data["bigram_freq"] <= max_bg_freq)]
		else:
			self.bgs = self.data
			
		if min_uni_freq > 1:
			print("    Selecting bigrams by unigram frequency")		
			self.bgs = self.bgs[(self.bgs["w1_freq"] >= min_uni_freq) & (self.bgs["w2_freq"] >= min_uni_freq)]
			
		if pos[0] != "" or pos[1] != "":	
			pos_1, pos_2 = pos
			pos_1 = compile(pos_1)
			pos_2 = compile(pos_2)
			print("    Selecting bigrams by POS")
			self.bgs = self.bgs[self.bgs["pos_1"].str.contains(pos_1) & self.bgs["pos_2"].str.contains(pos_2)]

		if words[0] != "" or words[1] != "":	
			w_1, w_2 = words
			w_1 = compile(w_1)
			w_2 = compile(w_2)
			print("    Selecting bigrams by words 1 & 2")
			self.bgs = self.bgs[self.bgs["w_1"].str.contains(w_1) & self.bgs["w_2"].str.contains(w_2)]
		
		total_ranges = []
		central_ranges = []
		
		for column in ["bigram_freq", "w1_freq", "w2_freq", "tp_d",
						"tp_b", "log_lklhd", "dice", "t_score", "z_score", 
						"mi_score", "mi3_score", "g_score", "delta_p12", "delta_p21"]:
			print(column)			
			scores = self.bgs[column].values
			
			lower = amin(scores)
			upper = amax(scores)
			lower_central, upper_central = percentile(scores, [(100-percent)/2, ((100-percent)/2)+percent])

			total_ranges.append([column, lower, upper])
			central_ranges.append([column, lower_central, upper_central])
		
		total_ranges = pd.DataFrame(total_ranges, columns=["score", "min", "max"])
		central_ranges = pd.DataFrame(central_ranges, columns=["score", str((100-percent)/2), str(((100-percent)/2)+percent)])
		return([total_ranges, central_ranges])
		
from search_settings import *
dt = Data()
total_ranges, central_ranges = dt.describe(num, cuts=cuts, seed=seed, disbalance_penalty = disbalance_penalty, words=[w_1, w_2], pos=[pos_1, pos_2], max_time=max_time, min_bg_freq=min_bg_freq, max_bg_freq=max_bg_freq, min_uni_freq=min_uni_freq, percent=95)

total_ranges.to_csv("data_ranges_total.csv", index=False)
central_ranges.to_csv("data_ranges_cental" + str(95) +"%.csv", index=False)