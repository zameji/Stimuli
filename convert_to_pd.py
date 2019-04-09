import pandas as pd
import numpy as np
import json

class Converter(object):
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

	def convert(self, outfile = "data.csv"):
	
		print("\nLoading the saved scores")
		print("    unigram frequency")
		with open("scores2/wfreqs.json", "r") as i:
			self.wfreq = json.loads(i.read())	
		
		print("    bigram frequency")
		with open("scores2/bigrams.json", "r") as i:
			self.bg_frq = json.loads(i.read())
							
		self.bgs = [x for x in self.bg_frq]
		self.compact = pd.DataFrame(np.transpose([self.bgs, [self.bg_frq[x] for x in self.bgs]]), columns = ["bigram", "bigram_freq"])
		del self.bg_frq
		
		self.compact["w1_freq"] = [self.wfreq[x.split()[0]] for x in self.bgs]
		self.compact["w2_freq"] = [self.wfreq[x.split()[1]] for x in self.bgs]
		del self.wfreq		
		
		print("    TP-D")	
		with open("scores2/fwd.json", "r") as i:
			self.tp_d = json.loads(i.read())	

		print("    TP-B")	
		with open("scores2/bckw.json", "r") as i:
			self.tp_b = json.loads(i.read())	
		
		self.compact["tp_d"] = [self.tp_d[x.split()[0]][x.split()[1]] for x in self.bgs]
		del self.tp_d
		self.compact["tp_b"] = [self.tp_b[x.split()[1]][x.split()[0]] for x in self.bgs]
		del self.tp_b
		
		print("    Log likelihood")	
		with open("scores2/llscore.json", "r") as i:
			self.log_lklhd = json.loads(i.read())	

		print("    Modified dice")	
		with open("scores2/dicescore.json", "r") as i:
			self.dice = json.loads(i.read())	
			
		self.compact["log_lklhd"] = [self.log_lklhd[x] for x in self.bgs]
		del self.log_lklhd
		self.compact["dice"] = [self.dice[x] for x in self.bgs]
		del self.dice
		
		print("    t-score")	
		with open("scores2/tscore.json", "r") as i:
			self.t_score = json.loads(i.read())	

		print("    z-score")	
		with open("scores2/zscore.json", "r") as i:
			self.z_score = json.loads(i.read())

		
		self.compact["t_score"] = [self.t_score[x] for x in self.bgs]
		del self.t_score
		self.compact["z_score"] = [self.z_score[x] for x in self.bgs]
		del self.z_score
		
		print("    delta_p-12")
		with open("scores2/delta_p12.json", "r") as i:
			self.delta_p12 = json.loads(i.read())	
			
		print("    delta_p-21")
		with open("scores2/delta_p21.json", "r") as i:
			self.delta_p21 = json.loads(i.read())					
		
		
		self.compact["delta_p12"] = [self.delta_p12[x] for x in self.bgs]
		del self.delta_p12
		self.compact["delta_p21"] = [self.delta_p21[x] for x in self.bgs]
		del self.delta_p21
		

		print("    MI-score")	
		with open("scores2/miscore.json", "r") as i:
			self.mi_score = json.loads(i.read())	
			
		print("    MI3-score")	
		with open("scores2/mi3score.json", "r") as i:
			self.mi3_score = json.loads(i.read())	

		print("    G-score")	
		with open("scores2/gscore.json", "r") as i:
			self.g_score = json.loads(i.read())	
									
		
		self.compact["mi_score"] = [self.mi_score[x] for x in self.bgs]
		del self.mi_score
		self.compact["mi3_score"] = [self.mi3_score[x] for x in self.bgs]
		del self.mi3_score
		self.compact["g_score"] = [self.g_score[x] for x in self.bgs]
		del self.g_score		
	
		
		print(self.compact.columns)
		self.compact[["tp_d", "tp_b", "log_lklhd", "dice", "t_score", "z_score", "mi_score", "mi3_score", "g_score", "delta_p12", "delta_p21"]] = self.compact[["tp_d", "tp_b", "log_lklhd", "dice", "t_score", "z_score", "mi_score", "mi3_score", "g_score", "delta_p12", "delta_p21"]].astype("float32")
		self.compact[["bigram_freq", "w1_freq", "w2_freq"]]	= self.compact[["bigram_freq", "w1_freq", "w2_freq"]].astype("int32")
		self.compact.to_csv(outfile, index=False)
	