import email, math, os

def load_tokens(email_path):
    file = open(email_path, "r")
    msg = email.message_from_file(file)
    return [x for line in email.iterators.body_line_iterator(msg) for x in line.split() ]

def log_probs(email_paths, smoothing_uni, smoothing_bi):
    d = {}
    totalWords, totalPairs = 0, 0
    d["<LENGTH LESS 8>"], d["<LENGTH 8 to 20>"], d["<LENGTH MORE 20>"] = 0, 0, 0
    for path in email_paths:
        l = load_tokens(path)
        for i, w in enumerate(l):
            # bigram feature if not last character
            if i != len(l) - 1:
                pair = (w, l[i+1]) # add the word and immediate next
                d[pair] = d.get(pair, 0) + 1
                totalPairs += 1
            #always count actual word for unigram feature
            d[w] = d.get(w, 0) + 1     
            totalWords += 1
            if len(w) < 8: # length < 8
                d["<LENGTH LESS 8>"] += 1
            elif len(w) >= 8 and len(w) < 20: # 8 <= len < 20
                d["<LENGTH 8 to 20>"] += 1
            else:   # len >= 20
                d["<LENGTH MORE 20>"] += 1   
    vSize = len(d) + 1
    for key in d:
        if key != "<LENGTH LESS 8>" and key != "<LENGTH 8 to 20>" and key != "<LENGTH MORE 20>":
            if len(key) == 1:
                d[key] = math.log((d.get(key) + smoothing_uni) / (totalWords + vSize * smoothing_uni))
            else:
                d[key] = math.log((d.get(key) + smoothing_bi) / (totalPairs + vSize * smoothing_bi))            
    d["<UNK UNI>"] = math.log(smoothing_uni / (totalWords + vSize * smoothing_uni))
    d["<UNK BI>"] = math.log(smoothing_bi / (totalPairs + vSize * smoothing_bi))
    d["<LENGTH LESS 8>"] = math.log((d.get("<LENGTH LESS 8>") + smoothing_uni) / (totalWords + vSize * smoothing_uni))
    d["<LENGTH 8 to 20>"] = math.log((d.get("<LENGTH 8 to 20>") + smoothing_uni) / (totalWords + vSize * smoothing_uni))
    d["<LENGTH MORE 20>"] = math.log((d.get("<LENGTH MORE 20>") + smoothing_uni) / (totalWords + vSize * smoothing_uni))
    return d

class SpamFilter(object):
    def __init__(self, spam_dir, ham_dir):
        smoothing_uni, smoothing_bi = 1e-15, 1e-9
        ham_paths = [ham_dir + '/' + x for x in os.listdir(ham_dir)]
        spam_paths = [spam_dir + '/' + x for x in os.listdir(spam_dir)]
        self.d_ham= log_probs(ham_paths, smoothing_uni, smoothing_bi)
        self.d_spam = log_probs(spam_paths, smoothing_uni, smoothing_bi)
        self.p_spam = len(spam_paths) / (len(spam_paths) + len(ham_paths) + 0.0)
        self.p_not_spam = 1.0 - self.p_spam
    
    def is_spam(self, email_path):
        spam_sum = 0        
        ham_sum = 0
        l = load_tokens(email_path)
        for i, w in enumerate(l):
            if w in self.d_spam: #unigram check
                spam_sum += self.d_spam[w]
            else:
                spam_sum += self.d_spam["<UNK UNI>"]
            if w in self.d_ham:
                ham_sum += self.d_ham[w]
            else:
                ham_sum += self.d_ham["<UNK UNI>"]   
            
            if len(w) < 8: # length < 8
                spam_sum += self.d_spam["<LENGTH LESS 8>"]
                ham_sum += self.d_ham["<LENGTH LESS 8>"]
            elif len(w) >= 8 and len(w) < 20: # 8 <= len < 20
                spam_sum += self.d_spam["<LENGTH 8 to 20>"]
                ham_sum += self.d_ham["<LENGTH 8 to 20>"]
            else:   # len >= 20
                spam_sum += self.d_spam["<LENGTH MORE 20>"]
                ham_sum += self.d_ham["<LENGTH MORE 20>"]
            if i != len(l) - 1: #bigram check as long as not last character
                pair = (w, l[i+1])
                if pair in self.d_spam:
                    spam_sum += self.d_spam[pair]
                else:
                    spam_sum += self.d_spam["<UNK BI>"]
                if pair in self.d_ham:
                    ham_sum += self.d_ham[pair]
                else:
                    ham_sum += self.d_ham["<UNK BI>"]                          
        return spam_sum + math.log(self.p_spam) > ham_sum + math.log(self.p_not_spam)

# accuracy testing script on dev files
#sf = SpamFilter("data/train/spam", "data/train/ham")
#paths = ["data/dev/spam/dev%d" % i for i in range(201, 401)]
#count = 0
#for p in paths:
#    if sf.is_spam(p):
#        count += 1
#    else:
#        print p
#print count / 200.0
#paths = ["data/dev/ham/dev%d" % i for i in range(1, 201)]
#count = 0
#for p in paths:
#    if not sf.is_spam(p):
#        count += 1
#    else:
#        print p
#print count / 200.0








