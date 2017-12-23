
# coding: utf-8

# # Program 1

# In[1]:

import numpy as np
import pandas as pd
import scipy.sparse as sp
import re
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from pyparsing import anyOpenTag, anyCloseTag
from xml.sax.saxutils import unescape as unescape


# In[2]:

# read in the dataset
with open("data/train.dat", "r") as tr:
    docs2 = tr.readlines()
with open("data/test.dat", "r") as te:
    docs3 = te.readlines()

#print len(docs1)
cls = []
reviewsdocs2 = []
reviews1 = []
for x in docs2:
    cls.append(x[0:2])
    reviewsdocs2.append(x[3:])
reviews = reviewsdocs2 + docs3
#print cls[0]
#print reviews[0]
#print len(docs2)
#print len(docs1)
#print len(reviews)


# In[3]:

#removing html tag
unescape_xml_entities = lambda k: unescape(k, {"&apos;": "'", "&quot;": '"', "&nbsp;":" "})

stripper = (anyOpenTag | anyCloseTag).suppress()
for i in range(0, len(reviews)):
    #print ("in loop")
    reviews1.append(unescape_xml_entities(stripper.transformString(reviews[i])))
#print reviews1[0]
print len(reviews1)
    



# In[3]:

#print reviews1[0]
r"""removing special characters from all the reviews"""
reviewwords = []
reviewwords1 = []
#reviews1 = []
for i in range(0, 50000):
    reviews1[i] = re.sub('[^a-zA-Z0-9]', ' ', reviews1[i])
    
#print reviews1[0]
#print (reviewwords[0])   
r"""splitting the lines into words and changing to lowercase letters"""
for i in range(0, 50000):
    reviewwords.append(reviews1[i].split())
    for j in range(0, len(reviewwords[i])):
        reviewwords[i][j] = reviewwords[i][j].lower()
#print reviewwords[0]
r"""removing words which have less than specified length"""
def removeLesLenWords(docs, len_min):
    #doc_new = []
    docs_new = []
    for d in docs:
        doc_new = []
        for word in d:
            if len(word) >= len_min :
                doc_new.append(word)
        docs_new.append(doc_new)
    return docs_new
        
reviewwords1 = removeLesLenWords(reviewwords, 4)


#for i in range(0, 25000):
#    for j in range(0, len(reviewwords[i])):
#        reviewwords[i][j] = re.sub('[^a-zA-Z0-9]', ' ', reviewwords[i][j])
#print (reviewwords[0])    

    
#print ("iam done")
#print len(reviewwords1)
#print (reviewwords1[0])


    


# In[5]:

def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )
        
def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat


# In[6]:

csrmat1 = build_matrix(reviewwords1)
csr_info(csrmat1)
#print csrmat1.shape
#csrmat1_v3 = csrmat1.toarray()


# In[ ]:




# In[7]:

csrmat1_2 = csr_idf(csrmat1, copy=True)
csrmat2 = csr_l2normalize(csrmat1_2, copy=True)
#print ("done here")


# In[160]:

train1 = csrmat2[:25000, :]
test1 = csrmat2[25000:, :] 
#test3 = csrmat2[46000:50000, :]
print ("done here")


# In[161]:

print train1.shape
print test1.shape
#print test3.shape


# In[162]:

#finding cosine similarity
#cossim = test3.dot(train1.T)
cossim = test1.dot(train1.T)


# In[164]:

#print cossim.shape[0]
simlist = []
simlist = cossim.toarray()
#print cossim.toarray()
#print ("in the middle")
#print simlist[0]
#print len(simlist)


# In[165]:

#print simlist[900]
xlen = len(simlist)
k = 150
#x = simlist[900]
#y = np.argsort(x)
#z= x[y]
#z = []
#print z
#print z[23000]
#print x[23000]
test_cls = []

for i in range (0, xlen):
    x = simlist[i]
    y = np.argsort(simlist[i])
    #z.append(x[y])
    pos_count = 0
    neg_count = 0
    ylen = len(y)
    z = y[ylen-k:]
    zlen = len(z)
    for j in range(0, zlen):
        if cls[z[j]]== '+1' :
            pos_count = pos_count+1
        else:
            neg_count = neg_count+1
    if pos_count > neg_count:
        test_cls.append('+1')
    else:
        test_cls.append('-1')

        


    
    


# In[166]:

test = open("data/result.dat", "w")
for i in range(0, len(test_cls)):
    test.write(test_cls[i])
    test.write('\n')
test.close()
    


# In[ ]:



