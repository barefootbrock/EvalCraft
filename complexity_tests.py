import glob
import string
import sys
import os
import random
import datetime
import time
import networkx as nx
import matplotlib.pyplot as plt

from base_classes import Dataset, Document
from eval_sumkeys import DATASET

from systems.textstar import Textstar


class Contrived(Dataset):
    has_sums = False
    has_kwds = False

    def __init__(self, sentCount=10, wordCount=25, charCount=8):
        self.sentCount = sentCount
        self.wordCount = wordCount
        self.charCount = charCount

        self.count = 10

    def __str__(self):
        return "Contrived dataset"
    
    def __iter__(self):
        while True:
            yield ContrivedDoc(
                self.sentCount,
                self.wordCount,
                self.charCount
            )


class ContrivedDoc(Document):
    has_summary = False
    has_key_words = False
    docNumber = 0
    
    def __init__(self, sentCount=100, wordCount=17, charCount=6):
        self.sentCount = sentCount
        self.wordCount = wordCount
        self.charCount = charCount

        self.docNumber += 1
        self.name = "contrived" + str(self.docNumber)
        self.fname = self.name + ".txt"

        self.text = self.createText()

    def createWord(self):
        return ''.join(random.choices(string.ascii_lowercase, k=self.charCount))

    def createSent(self):
        return ' '.join(self.createWord() for _ in range(self.wordCount)).capitalize() + '.'

    def createText(self):
        return ' '.join(self.createSent() for _ in range(self.sentCount))
    
    def __str__(self) -> str:
        return "Contrived document (id: %s)" % self.docNumber
    
    def as_text(self):
        return self.text
    
    def summary(self):
        pass
    
    def key_words(self):
        pass


# SETTINGS ------------------------------------------------

# number of keyphrases and summary sentences
wk, sk = 5, 5 # best for Textstar key phrase extraction

# Prevent printing from summarization code
quiet = True

# choice of NLP summarizer and key-word extractor
SYSTEM = Textstar(
    stanza_path="/Users/brockfamily/Documents/UNT/StanzaGraphs/",
    ranker=nx.degree_centrality,
    trim=98
)

# choice of dataset
DATASET = Contrived(
    charCount=3
)

# SETTINGS ------------------------------------------------



def avg(xs):
    s = sum(xs)
    l = len(xs)
    if 0 == l:
        return None
    return s/l


def printProgress(progress, width=100, end="", same_line=True):
    count = round(progress * width)
    line_end = '\r' if same_line else '\n'
    print("|" + "="*count + " "*(width-count) +
          "| %.1f%%" % (progress * 100), end, end=line_end)


class Quiet():
    """Prevent printing. Usage:
    with Quiet():
      run_and_print_a_lot(args) #Nothing will print
    """

    def __enter__(self):
        self.devnull = open(os.devnull, "w")
        self.original = sys.stdout
        sys.stdout = self.devnull

    def __exit__(self, type, value, traceback):
        sys.stdout = self.original
        self.devnull.close()

def runAndTime(system, dataset, iterations):
    runTimes = []

    for i, document in enumerate(dataset):
        if i >= iterations:
            break

        # Try to summarize the document
        docStartTime = time.time()
        if quiet:
            with Quiet():
                keys, exabs = system.process_text(
                    document,
                    summarize=True,
                    key_words=True,
                    sum_len=sk,
                    kwds_len=wk
                )

        else:
            keys, exabs = system.process_text(
                document,
                summarize=True,
                key_words=True,
                sum_len=sk,
                kwds_len=wk
            )
        
        runTimes.append(time.time() - docStartTime)
    
    return avg(runTimes)

def evaluate(system, dataset):
    print()

    x = []
    y = []
    #Pre run to remove difference in first run
    runAndTime(system, dataset, 25)

    for i in range(10, 110, 10):
        dataset.charCount = i
        y.append(runAndTime(system, dataset, 10))
        x.append(i)
        print(x[-1], y[-1])

    plt.plot(x, y)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()


if __name__ == '__main__':
    evaluate(
        SYSTEM, DATASET
    )
