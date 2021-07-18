import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = stopwords.words("english")
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')


def reminder(imperatives):
    reminder_list = []
    for i in imperatives[:10]:
        # for i in ["Take water, comfortable shoes and cloths for your adventure."]:
        conjunctions = []
        result = dependency_parser.raw_parse(i)

        dep = result.__next__()
        print("sentence:-", i)
        reminder_list.append([])

        for j in list(dep.triples()):
            temp1 = BAD_SYMBOLS_RE.sub(' ', j[2][0])
            temp2 = REPLACE_BY_SPACE_RE.sub(' ', j[2][0])
            # print(j)
            if j[1] == "dobj" and j[2][0].lower() not in STOPWORDS and temp1 == j[2][0] and temp2 == j[2][0] and j[2][
                1] in ["NN", "NNS"]:
                reminder_list[-1].append([j[0][0].lower(), j[2][0].lower()])
                for k in list(dep.triples()):
                    if k[0][0].lower() == j[2][0].lower() and k[2][1].lower() == "jj":
                        reminder_list[-1][-1] = [j[0][0].lower(), k[2][0].lower() + " " + j[2][0].lower()]

            if j[1] == "cc":
                conjunctions.append(j)
        if len(reminder_list[-1]) > 1 and len(conjunctions) != 0:
            if len(reminder_list[-1]) <= len(conjunctions):
                for i in range(len(reminder_list[-1]) - 1):
                    reminder_list[-1].insert(2 * i + 1, [conjunctions[i][2][0].lower()])
            else:
                for i in range(len(conjunctions)):
                    reminder_list[-1].insert(2 * i + 1, [conjunctions[i][2][0].lower()])

        if len(reminder_list[-1]) == 0:
            del reminder_list[-1]
            print("NOTHING")
            continue
        temp = ""
        for i in reminder_list[-1]:
            for j in i:
                temp += j + " "
        reminder_list[-1] = temp
        print("stanford_parser:-", temp)

    return reminder_list