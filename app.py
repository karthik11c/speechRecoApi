from flask import Flask, render_template,request,jsonify
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from collections import Counter

#import pandas as pd
#import numpy as np

# Configure the app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route('/')
def hello_world():
    """
    Serves the main template file.
    You probably do NOT want to modify this.
    """
    return render_template('index.html')

@app.after_request
def add_header(r):
    """
    Disable all caching.
    You probably do NOT want to modify this.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

#####################################
# REST backend interface


#@app.route('/data/')
#def get_data():
#    """
#    Example GET endpoint to get data from the backend.
#
#    """
#    data = pd.DataFrame(np.random.randn(6,4))
#    return data.to_csv()

#textsearch api
@app.route('/process',methods=['GET', 'POST'])
def get_results():
    transcript_text = request.args.get('rawText')
    all_stopwords = stopwords.words('english')
    all_stopwords.append('show')
    all_stopwords.append('want')
    all_stopwords.append('buy')
    all_stopwords.append('see')
    all_stopwords.append('wish')
    all_stopwords.append('display')
    all_stopwords.append('some')
    all_stopwords.append('please')
    all_stopwords.append('list')

    def load_text(filename): 
        file = open(filename,'r')
        text = file.read()
        file.close()
        return text
    
    def load_description(doc):
        mapping = dict()
        #one entry in each line
        for line in doc.split('\n'):
            tokens = line.split()
            #if no of tokens less than 2 --> incorrect desc
            if len(line)<2:
                continue
            #first -- id rest -- desc
            image_id, image_desc = tokens[0], tokens[1:]
            image_id = image_id.split('.')[0]
            
            #convert description token back to string
            image_desc = ' '.join(image_desc)
            #create a list (containing all desc of a given image)
            if image_id not in mapping:
                mapping[image_id] = list()
            mapping[image_id].append(image_desc)
        return mapping

    #Cleaning description -- convert to lowercase, remove punctuation, remove words less than some len, remove words with number
    def clean_description(description):
        #remove punctuation -- make translation table
        #param1 - to be replaced by param2 ---- param3 removed
        table = str.maketrans('','',string.punctuation)
        
        for key, desc_list in descriptions.items():
            for i in range(len(desc_list)):
                desc = desc_list[i]
                #tokenize
                desc = desc.split()
                #to lower
                desc = [word.lower() for word in desc]
                #remove punctuation
                desc = [word.translate(table) for word in desc]
                #remove words less in len
                desc = [word for word in desc if len(word)>1]
                #remove numbers
                desc = [word for word in desc if word.isalpha()]
                #re-convert to desc
                desc_list[i] =  ' '.join(desc)
    
    def save_description(description, filename):
        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        data = ('\n').join(lines)
        file = open(filename,'w')
        file.write(data)
        file.close()    

    #path = '\static\wallpapers\'
    import os
    rootpath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(rootpath, 'static/wallpapers/')
    doc = load_text(path + 'descriptions.txt')
    
    # parse descriptions
    descriptions = load_description(doc)
    print('Loaded: %d ' % len(descriptions))
    
    # clean descriptions
    clean_description(descriptions)
    # summarize vocabulary
    #vocabulary = to_vocabulary(descriptions)
    
    #print('Vocabulary Size: %d' % len(vocabulary))
    
    descrOut = path + 'clean_descriptions.txt'
    
    save_description(descriptions, descrOut)
    
    def load_doc(filename):
        file = open(filename,'r')
        text = file.read()
        file.close()    
        return text
    
    #Image Identifier --- TrainImage.txt
        #2513260012_03d33305cf.png
    def load_set(filename):
        doc = load_doc(filename)
        dataset = list()
        
        for line in doc.split('\n'):
            if len(line)<1:
                continue
            identifier = line.split('.')[0]
            dataset.append(identifier)
        return set(dataset)
    
    #load cleaned description
    #1000268201_693b08cb0e child in pink dress is climbing up set of stairs in an entry way
    #assign start and end token to the description
    def load_clean_descriptions(filename, dataset):
        doc = load_doc(filename)
        descriptions = dict()
        
        for line in doc.split('\n'):
            tokens = line.split()
            image_id, image_desc = tokens[0],tokens[1:]
            #if image_id not in dataset ignore
            if image_id in dataset:
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                #desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                desc = ' '.join(image_desc) 
                descriptions[image_id].append(desc)
        return descriptions
    
    #user_description = 'yellow white wallpaper'
    user_description = transcript_text#'Please show me a list of red texture wallpaper' #user input in text 
    user_description = user_description.lower()
    
    table = str.maketrans('','',string.punctuation)
    #desc = user_description.split()
    
    text_tokens = word_tokenize(user_description)
    desc = [word for word in text_tokens if not word in all_stopwords]
    #desc = [word.lower() for word in desc]
    desc = [word.translate(table) for word in desc]
    desc = [word for word in desc if len(word)>1]
    desc = [word for word in desc if word.isalpha()]
    user_description =  ' '.join(desc)
    print(user_description)
    
    imageName = path + 'image_names.txt'
    imagesLabel = load_set(imageName)
    test_descriptions = load_clean_descriptions(descrOut, imagesLabel)
    
    print(test_descriptions)
    
    from nltk.translate.bleu_score import sentence_bleu
    
    matchedFiles = set()
    score = []
    img_id =[]
    Z= []
        
    words_list = user_description.split() # user descrip  
    
    word_freq = {}
    
    for k,v in test_descriptions.items():
      word_freq[k] = ''.join(v).split()
    
    #print(test_descriptions)  
    print(word_freq)
    print(words_list)
    
    keys = []
    for x in words_list:
      list_of_keys  = [key for key, list_of_values in word_freq.items() if x in list_of_values]
    
      if list_of_keys:
        keys.extend(list_of_keys)
      else:
        print("Value does not exist in the dictionary")

    def most_frequent(keys):
        occurence_count = Counter(keys)
        return occurence_count.most_common(3)
    
    print("3 matching image tags in the format (tag, freq) :")
    print(most_frequent(keys))
    
    key_freq = most_frequent(keys)
    image_names = []
    
    for i in key_freq:
      image_names.append(i[0])
    
    print(image_names)
    
    from shutil import copyfile
    import glob
    import os
    
    dir = path + 'output_images/'
    
    filelist = glob.glob(os.path.join(dir, "*"))
    print(filelist)
    
    for f in filelist:
        os.remove(f)
    
    i=0
    for img in image_names:
        img_path = path + img + '.png'
        i += 1
        #matched_img_file.write(descriptions[img][0]+ '\\n')
        copyfile(img_path, path + 'output_images/' + img + '.png')    
        print(path + 'output_images/' + img + '.png')

    print(image_names)
    if not image_names:
        return jsonify([])
    return jsonify(image_names)
    #return render_template("index.html", link=image_names)

if __name__ == '__main__':
    app.run(debug=True)

