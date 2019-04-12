import re
import tqdm
from collections import Counter
import os
import json
import sys
import gc
gc.enable()

"""
This script counts the dispersion of bigram accross texts - within the corpus as a whole and for each COCA genre individually. 
Then it normalizes the dispersion counts by the file counts in each subset.
"""

class DispersionCounter(object):

	def __init__(self, path="C:/projects/COCA"):
	
		self.files = []
		for dirpath, dirnames, filenames in os.walk(path):
			self.files.extend([os.path.join(dirpath, file) for file in filenames])
		
		self.dispersion = Counter()
		self.total = 0
		
		self.log = open("log.txt", "w+")

		with open("C:/projects/Stimuli/scores2/bigrams.json", "r") as fin:
			self.relevants = json.loads(fin.read())
			self.relevants = set([x for x in self.relevants])
	
	def preprocess(self, filename):

		with open(filename, "r") as i:
			doc = i.read()
			
		doc = re.sub("_", "", doc)
		doc = re.sub("\t+", "\t", doc)
		doc = doc.split("\n")
		doc = [word.split("\t") for word in doc]

		docs = []
		d = []
		accepted = re.compile("^[^@]+$")
		for word in doc:
			if len(word) == 3:
				d.append([word[0], word[2]])
				
			elif word[0].startswith("##"):
				d = ["_".join(w).strip() for w in d]
				d = [x + " " + y for x,y in zip(d[0:-1], d[1:]) if not (x.endswith("_y") or y.endswith("_y"))]
				d = set([x for x in d if re.match(accepted, x) != None])
				docs.append(d)
				d = []
				
			else:
				pass
				
		d = ["_".join(w).strip() for w in d]
		d = [x + " " + y for x,y in zip(d[0:-1], d[1:]) if not (x.endswith("_y") or y.endswith("_y"))]	
		d = [x for x in d if re.match(accepted, x) != None]
		docs.append(set(d))
		return(docs)
	
	def collect(self, extension):
		total = 0
		dispersion = Counter()
		print("Collecting %s" % extension)
		dt = [filename for filename in self.files if re.search(extension, filename) != None]
		# dt = dt[0:2]
		for filename in tqdm.tqdm(dt):
		# for filename in dt:
			bgs = self.preprocess(filename)
			
			for bg in bgs:
				# print(len(bg))
				bg = bg.intersection(self.relevants)			
				dispersion.update(bg)
			
			total += len(bgs)

		for item in dispersion:
			dispersion[item] = (1.0*dispersion[item])/total

		with open("dispersion" + extension + ".json", "w+") as fout:
			fout.write(json.dumps(dispersion))

		self.log.write("Files in " + extension + ": "+str(total) + "\n")		
		gc.collect()
		
		iter = 0
		for item in dispersion:
			print("%s - %f" % (item, dispersion[item]))
			iter += 1
			if iter == 10:
				break
		
		self.dispersion.update(dispersion)
		self.total += total

	def save(self):
		for item in tqdm.tqdm(self.dispersion):
			self.dispersion[item] = (1.0*self.dispersion[item])/self.total

		with open("dispersion.json", "w+") as fout:
			fout.write(json.dumps(self.dispersion))	
			
		self.log.write("Total files: " + str(self.total))
		self.log.close()
		
worker = DispersionCounter()

for ext in tqdm.tqdm(["acad", "fic", "news", "mag", "spok"]):
# for ext in ["acad", "fic", "news", "mag", "spok"]:
	worker.collect(ext)
	gc.collect()
	
print("Collecting done - preprocessing final data")
worker.save()
		
sys.exit(0)