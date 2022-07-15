from base_classes import NLPSystem
import sys
import networkx as nx
import nltk


class Test(NLPSystem):
    def __init__(self, path):
        """
        Args:
          stanza_path: Path to the StanzaGraphs folder for importing purpose
        """
        self.path = path
        sys.path.append(path)
    
    def __str__(self):
        return "Test System at " + self.path

    def process_text(self, document, summarize=True, key_words=True, sum_len=5, kwds_len=5):
        from encavg import summarize

        text = document.as_text()

        try:
            summary = summarize(text, sumLen=sum_len)
        except:
            raise

        return [], summary
