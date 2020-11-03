import os, re

class Corpus(object):

    def __init__(self, path: str, extension='.txt', pattern='.*', recursive=True, in_memory=True):
        """An iterator factory, standardizing various supported corpora into an iterable of Documents
        objects.

        Args:
            path (str): path to the corpus files
            extension (str, optional): extension of the corpus files. Any extension will be accepted if empty string is provided. Defaults to '.txt'.
            pattern (str, optional): RegEx describing the names of the corpus files. Defaults to '.*'.
            recursive (bool, optional): Whether subfolders should be explored, too. Defaults to True.
            in_memory (bool, optional): Whether the whole corpus should be loaded into memory (True), or read/processed on demand. Defaults to True.
        """
           
        self.in_memory = in_memory
        self.files = []

        pattern = re.compile(pattern)
        for dirpath, dirnames, filename in os.walk(path):
            if (recursive or dirpath == path) and (re.match(pattern, filename) and filename.endswith(extension)):
                self.files.append(os.path.join(dirpath, filename))
                
        if self.in_memory:
            pass
            # process
        else:
            pass
            # peek at the first file to know the available data
        
    def __iter__(self):
        """Start iterating
        """
        self.pointer = 0
        return self
        
    def __next__(self):