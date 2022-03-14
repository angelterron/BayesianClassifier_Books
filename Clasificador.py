import glob
import re
import collections
import pickle

def analize_words(word, dict):
    if (word[-2:] is 'ch' or word[-2:] is 'sh') and (word[0:-2]+'es') in dict:
        word = word[0:-2]+'es'

    if (word[-1:] is 'o' or word[-1:] is 's' or word[-1:] is 'x' or word[-1:] is 'z' ) and (word[0:-1]+'es') in dict:
        word = word[0:-1]+'es'

    if word[-1:] is 'y' and (word[0:-1]+'ies') in dict:
        word = word[0:-1]+'ies'

    if word[-2:] is 'fe' and word[0:-2]+'ves' in dict:
        word = word[0:-2]+'ves'

    if word[-1:] is 'f' and word[0:-1]+'ves' in dict:
        word = word[0:-1]+'ves'

    if (word + 's') in dict:
        word = (word + 's')

    return word


commonWords = ['the','and', 'that', 'are', 'for', 'with'
                , 'from', 'this', 'can', 'these', 'not'
                , 'other' , 'their', 'into', 'which'
                , 'have', 'one', 'they', 'also', 'has'
                , 'two', 'but', 'some', 'more', 'all'
                , 'when', 'each', 'such', 'chapter'
                , 'figure', 'its', 'many' , 'may', 'between'
                , 'most', 'will', 'been', 'see', 'was'
                , 'would', 'any', 'what', 'there', 'than'
                , 'were', 'your', 'yours', 'our' , 'ours', 'you']


classification = ['biology', 'business', 'law', 'literature', 'math', 'social_sciences']
dict = {'biology': {},'business': {}, 'law' : {}, 'literature' : {} , 'math' : {}, 'social_sciences' : {}}
confusionMatrix = {'biology': {},'business': {}, 'law' : {}, 'literature' : {} , 'math' : {}, 'social_sciences' : {}}
confusionMatrixMCC  = {'biology': {},'business': {}, 'law' : {}, 'literature' : {} , 'math' : {}, 'social_sciences' : {}}
totalWords = []
training = False
singleBook = True

if training :
    for c in classification:
        myFiles = glob.glob('D:\\MIA\\AA\\libros\\'+c+'\\*.txt')
        for f in myFiles[:24]:
            with open(f,'r', encoding="utf8") as file:
                for line in file:
                    for word in line.split():
                        word = re.sub('[^A-Za-z]+','',word).lower()

                        if(len(word) > 2) and (len(word) < 13) and word not in commonWords:

                            word = analize_words(word,dict[c])

                            if word in dict[c]:
                                dict[c][word] = dict[c][word] + 1
                            else:
                                dict[c][word] = 1
                                if word not in totalWords:
                                    totalWords.append(word)
                            if 'N' in dict[c]:
                                dict[c]['N'] = dict[c]['N'] + 1
                            else:
                                dict[c]['N'] = 1

    for c in classification:
        dict[c] = sorted(dict[c].items(), key=lambda kv: kv[1],reverse=True)
        dict[c] = collections.OrderedDict(dict[c])

    dict['T'] = len(totalWords)
    a_file = open("data.pkl", "wb")
    pickle.dump(dict, a_file)
    a_file.close()

elif singleBook is False:
    a_file = open("data.pkl", "rb")
    dict = pickle.load(a_file)
    a_file.close()

    for c in classification:
        myFiles = glob.glob('D:\\MIA\\AA\\libros\\'+c+'\\*.txt')
        for f in myFiles[24:]:
            with open(f,'r', encoding="utf8") as file:
                probabilityByClass = {'biology': 1,'business': 1, 'law' : 1, 'literature' : 1 , 'math' : 1, 'social_sciences' : 1}
                auxDict = {}
                for line in file:
                    for word in line.split():
                        word = re.sub('[^A-Za-z]+','',word).lower()

                        if(len(word) > 2) and (len(word) < 13) and word not in commonWords:
                            for cla in classification:
                                word = analize_words(word,auxDict)
                                if word in auxDict:
                                    auxDict[word] = auxDict[word] + 1
                                else:
                                    auxDict[word] = 1

            auxDict = sorted(auxDict.items(), key=lambda kv: kv[1],reverse=True)
            auxDict = collections.OrderedDict(auxDict[:10])
            for k in auxDict:
                for cla in classification:
                    if k in dict[cla]:
                        p = dict[cla][k]
                    else:
                        p = 0
                    probabilityByClass[cla] = probabilityByClass[cla] * ((p+1)/(dict[cla]['N']+dict['T']))

            for cla in classification:
                probabilityByClass[cla] = probabilityByClass[cla] * 0.16

            maxN = -1
            maxC = ''
            for k in probabilityByClass:
                if probabilityByClass[k] > maxN:
                    maxN = probabilityByClass[k]
                    maxC = k
            print(f +' '+c+' '+maxC)
            if maxC in confusionMatrix[c]:
                confusionMatrix[c][maxC] = confusionMatrix[c][maxC] + 1
            else:
                confusionMatrix[c][maxC] = 1

    aux1 = 0
    aux2 = 0
    sum = 0
    for c1 in classification:
        for c2 in classification:
            if c2 not in confusionMatrix[c1]:
                confusionMatrix[c1][c2] = 0

            aux2 = aux2 + confusionMatrix[c1][c2]
            if c1 == c2:
                aux1 = aux1 + confusionMatrix[c1][c2]
                confusionMatrixMCC[c1]['TP'] =  confusionMatrix[c1][c2]
            else:
                if 'FN' not in confusionMatrixMCC[c1]:
                    confusionMatrixMCC[c1]['FN'] = confusionMatrix[c1][c2]
                    confusionMatrixMCC[c1]['TN'] = confusionMatrix[c1][c2]
                else:
                    confusionMatrixMCC[c1]['FN'] = confusionMatrixMCC[c1]['FN'] + confusionMatrix[c1][c2]
                    confusionMatrixMCC[c1]['TN'] = confusionMatrixMCC[c1]['TN'] + confusionMatrix[c1][c2]

                if c1 not in confusionMatrix[c2]:
                    confusionMatrix[c2][c1] = 0

                if 'FP' not in confusionMatrixMCC[c1]:
                    confusionMatrixMCC[c1]['FP'] = confusionMatrix[c2][c1]
                    confusionMatrixMCC[c1]['TN'] = confusionMatrix[c2][c1]
                else:
                    confusionMatrixMCC[c1]['FP'] = confusionMatrixMCC[c1]['FP'] + confusionMatrix[c2][c1]
                    confusionMatrixMCC[c1]['TN'] = confusionMatrixMCC[c1]['TN'] + confusionMatrix[c2][c1]
            sum = sum + confusionMatrix[c2][c1]

    for c in classification:
        confusionMatrixMCC[c]['TN'] = sum - confusionMatrixMCC[c]['TN'] -confusionMatrixMCC[c]['TP']

    for c in classification:
        print(c)
        print(dict[c])

    acc = aux1/aux2

    print('Matriz de confusion')
    print(confusionMatrix)
    print('Accuracy: '+str(acc))
    for c in classification:
        print(c)
        print('Precision: ' + str(confusionMatrixMCC[c]['TP']/(confusionMatrixMCC[c]['TP']+confusionMatrixMCC[c]['FP'])))
        print('Recall: ' + str(confusionMatrixMCC[c]['TP']/(confusionMatrixMCC[c]['TP']+confusionMatrixMCC[c]['FN'])))
else:
    a_file = open("data.pkl", "rb")
    dict = pickle.load(a_file)
    a_file.close()
    with open('D:\\MIA\\AA\\libros\\Social_sciences\\Soci_23.txt','r', encoding="utf8") as file:
        probabilityByClass = {'biology': 1,'business': 1, 'law' : 1, 'literature' : 1 , 'math' : 1, 'social_sciences' : 1}
        auxDict = {}
        for line in file:
            for word in line.split():
                word = re.sub('[^A-Za-z]+','',word).lower()

                if(len(word) > 2) and (len(word) < 13) and word not in commonWords:
                    for cla in classification:
                        word = analize_words(word,auxDict)
                        if word in auxDict:
                            auxDict[word] = auxDict[word] + 1
                        else:
                            auxDict[word] = 1

    auxDict = sorted(auxDict.items(), key=lambda kv: kv[1],reverse=True)
    auxDict = collections.OrderedDict(auxDict[:10])
    for k in auxDict:
        for cla in classification:
            if k in dict[cla]:
                p = dict[cla][k]
            else:
                p = 0
            probabilityByClass[cla] = probabilityByClass[cla] * ((p+1)/(dict[cla]['N']+dict['T']))

    for cla in classification:
        probabilityByClass[cla] = probabilityByClass[cla] * 0.16

    maxN = -1
    maxC = ''
    for k in probabilityByClass:
        if probabilityByClass[k] > maxN:
            maxN = probabilityByClass[k]
            maxC = k
    print('Clasificador: '+maxC)
