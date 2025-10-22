import os
import logging
import numpy as np
import itertools
from typing import List
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.svm import SVC
from src.config import Config
from src.utils import Utils
class Metrics:
    @staticmethod
    def tc(doc_list_str:List[str],vocab,top_words_list_str:List[str],cv_type='c_v'):
        top_words_list=Utils.split_list_str(top_words_list_str)
        num_top_words=len(top_words_list)
        for top_words in top_words_list:
            assert len(top_words)==num_top_words
        doc_list=Utils.split_list_str(doc_list_str)
        dictionary=Dictionary(Utils.split_list_str(vocab))
        cm=CoherenceModel(texts=doc_list,dictionary=dictionary,
                                       topics=top_words_list,topn=num_top_words,coherence=cv_type)
        cv_per_topic=cm.get_coherence_per_topic()
        score=np.mean(cv_per_topic)
        return cv_per_topic,score
    @staticmethod
    def tc_on_wiki(use_kaggle,use_colab,top_words_path,jar_dir=Config.JAR_DIR,cv_type='C_V'):
        random_number=np.random.randint(100000) # @QUESTION: what does this temp file save?
        if use_kaggle:
            wiki_dir=Config.WIKI_DIR_KAGGLE
        elif use_colab:
            wiki_dir=Config.WIKI_DIR_COLAB
        else:
            wiki_dir=Config.WIKI_DIR_LOCAL+'/'
        os.system(f"java -jar {os.path.join(jar_dir, 'pametto.jar')} {wiki_dir} {cv_type} {top_words_path} > tmp_{random_number}.txt")
        cv_score=[]
        with open(f'tmp_{random_number}.txt','r') as f:
            for line in f.readlines():
                if not line.startswith('202'): # If not error
                    cv_score.append(float(line.strip().split()[1]))
        os.remove(f'tmp_{random_number}.txt')
        return cv_score,sum(cv_score)/len(cv_score)
    @staticmethod
    def td(top_words_list_str:List[str]):
        K=len(top_words_list_str)
        T=len(top_words_list_str[0].split())
        vectorizer=CountVectorizer(tokenizer=lambda x:x.split()) # @QUESTION: lambda?
        counter=vectorizer.fit_transform(top_words_list_str).toarray()
        TF=counter.sum(axis=0)
        TD=(TF==1).sum()/(K*T)
        return TD
    @staticmethod
    def irbo(top_words_list_str:List[str],topk:int=10,weight:float=0.9):
        top_words_list = Utils.split_list_str(top_words_list_str)
        return RBO.irbo(top_words_list, topk=topk, weight=weight)
    
    # Clustering:
    @staticmethod
    def nmi(theta,labels):
        preds=np.argmax(theta,axis=1)
        return metrics.cluster.normalized_mutual_info_score(labels_true=labels,labels_pred=preds)
    @staticmethod
    def purity(theta,labels):
        preds=np.argmax(theta,axis=1)
        contingency_matrix=metrics.cluster.contingency_matrix(labels_true=labels,labels_pred=preds)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    @staticmethod
    def inverse_purity(theta,labels):
        preds=np.argmax(theta,axis=1)
        contingency_matrix=metrics.cluster.contingency_matrix(labels_true=labels,labels_pred=preds)
        recall = contingency_matrix / contingency_matrix.sum(axis=1).reshape(-1, 1)
        inverse_purity = (np.amax(recall, axis=1) * contingency_matrix.sum(axis=1)).sum() / contingency_matrix.sum()
        return inverse_purity
    @staticmethod
    def harmonic_purity(theta,labels):
        preds=np.argmax(theta,axis=1)
        contingency_matrix=metrics.cluster.contingency_matrix(labels_true=labels,labels_pred=preds)
        precision = contingency_matrix / contingency_matrix.sum(axis=0).reshape(1, -1)
        recall = contingency_matrix / contingency_matrix.sum(axis=1).reshape(-1, 1)
        
        # Handle division by zero: replace inf/nan with 0
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        
        # Calculate F1, avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1)
        
        harmonic_purity = (np.amax(f1, axis=1) * contingency_matrix.sum(axis=1)).sum() / contingency_matrix.sum()
        return harmonic_purity
    @staticmethod
    def ari(theta,labels):
        preds=np.argmax(theta,axis=1)
        return metrics.adjusted_rand_score(labels_true=labels,labels_pred=preds)
    @staticmethod
    def mis(theta,labels):
        preds=np.argmax(theta,axis=1)
        return metrics.normalized_mutual_info_score(labels_true=labels,labels_pred=preds)
    
    # Classification:
    @staticmethod
    def accuracy(train_theta,test_theta,train_labels,test_labels,classifier='SVM',gamma='scale',tune=False):
        if tune:
            acc=0
            logger=logging.getLogger('main')
            for C in [0.1, 1, 10, 100, 1000]:
                for gamma in ['scale', 'auto', 10, 1, 0.1, 0.01, 0.001]:
                    for kernel in ['rbf', 'linear']:
                        logger.info(f'C: {C}, gamma: {gamma}, kernel: {kernel}')
                        if classifier == 'SVM':
                            clf = SVC(C=C, kernel=kernel, gamma=gamma)
                        else:
                            raise NotImplementedError('Classifier not supported')
                        clf.fit(train_theta,train_labels)
                        preds=clf.predict(test_theta)
                        this_acc=metrics.accuracy_score(y_true=test_labels,y_pred=preds)
                        acc=max(acc,this_acc)
                        logger.info(f'This accuracy: {this_acc}')
        else:
            if classifier=='SVM':
                clf=SVC(gamma=gamma)
            else:
                raise NotImplementedError('Classifier not supported')
            clf.fit(train_theta,train_labels)
            preds=clf.predict(test_theta)
            acc=metrics.accuracy_score(y_true=test_labels,y_pred=preds)
        return acc
    @staticmethod
    def marco_f1(train_theta,test_theta,train_labels,test_labels,classifier='SVM',gamma='scale',tune=False):
        if tune:
            marco_f1=0
            logger=logging.getLogger('main')
            for C in [0.1, 1, 10, 100, 1000]:
                for gamma in ['scale', 'auto', 10, 1, 0.1, 0.01, 0.001]:
                    for kernel in ['rbf', 'linear']:
                        logger.info(f'C: {C}, gamma: {gamma}, kernel: {kernel}')
                        if classifier == 'SVM':
                            clf = SVC(C=C, kernel=kernel, gamma=gamma)
                        else:
                            raise NotImplementedError('Classifier not supported')
                        clf.fit(train_theta,train_labels)
                        preds=clf.predict(test_theta)
                        this_marco_f1=metrics.f1_score(y_true=test_labels,y_pred=preds,average='macro')
                        marco_f1=max(marco_f1,this_marco_f1)
                        logger.info(f'This accuracy: {this_marco_f1}')
        else:
            if classifier=='SVM':
                clf=SVC(gamma=gamma)
            else:
                raise NotImplementedError('Classifier not supported')
            clf.fit(train_theta,train_labels)
            preds=clf.predict(test_theta)
            marco_f1=metrics.f1_score(y_true=test_labels,y_pred=preds,average='macro')
        return marco_f1
    
class RBO:
    """Rank-Biased Overlap implementation for topic diversity measurement."""
    
    @staticmethod
    def set_at_depth(lst, depth):
        ans = set()
        for v in lst[:depth]:
            if isinstance(v, set):
                ans.update(v)
            else:
                ans.add(v)
        return ans

    @staticmethod
    def raw_overlap(list1, list2, depth):
        set1, set2 = RBO.set_at_depth(list1, depth), RBO.set_at_depth(list2, depth)
        return len(set1.intersection(set2)), len(set1), len(set2)

    @staticmethod
    def overlap(list1, list2, depth):
        return RBO.agreement(list1, list2, depth) * min(depth, len(list1), len(list2))

    @staticmethod
    def agreement(list1, list2, depth):
        len_intersection, len_set1, len_set2 = RBO.raw_overlap(list1, list2, depth)
        return 2 * len_intersection / (len_set1 + len_set2)

    @staticmethod
    def rbo_ext(list1, list2, p):
        """RBO point estimate based on extrapolating observed overlap."""
        S, L = sorted((list1, list2), key=len)
        s, l = len(S), len(L)
        x_l = RBO.overlap(list1, list2, l)
        x_s = RBO.overlap(list1, list2, s)
        sum1 = sum(p ** d * RBO.agreement(list1, list2, d) for d in range(1, l + 1))
        sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
        term1 = (1 - p) / p * (sum1 + sum2)
        term2 = p ** l * ((x_l - x_s) / l + x_s / s)
        return term1 + term2

    @staticmethod
    def irbo(top_words, topk=10, weight=0.9):
        """Calculate Inverted Rank-Biased Overlap (IRBO) of top words.
        
        Args:
            top_words: List of lists, each containing top words for a topic
            topk: Number of top words to consider
            weight: RBO parameter p (probability of continuing)
        
        Returns:
            IRBO score (1 - mean RBO)
        """
        if not top_words:
            raise ValueError("top_words cannot be empty.")

        min_len = min(len(lst) for lst in top_words)
        if topk > min_len:
            raise ValueError(f"topk={topk} > min_len({min_len}).")

        scores = []
        for list1, list2 in itertools.combinations(top_words, 2):
            rbo_val = RBO.rbo_ext(list1[:topk], list2[:topk], p=weight)
            scores.append(rbo_val)

        return 1.0 - np.mean(scores)