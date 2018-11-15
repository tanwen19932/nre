import numpy as np


def read_all_lines(path):
    all_lines = []
    with open(path, 'r', encoding='utf-8') as file:
        temp_lines = file.readlines()
        for line in temp_lines:
            line = line.strip()
            if line:
                all_lines.append(line)
    return all_lines

def load_raw(path1, path2):
    al_train = read_all_lines(path1)
    al_test = read_all_lines(path2)
    train_raw = []
    test_raw = []
    i = 0
    for line in al_train:
        train_raw.append(line.split('|'))
        i = i + 1
    i = 0
    for line in al_test:
        test_raw.append(line.split('|'))
        i = i + 1
    return train_raw, test_raw


def label2index(label):
    if label == 'Other':
        return 0
    if label == 'Cause-Effect':
        return 1
    if label == 'Component-Whole':
        return 2
    if label == 'Content-Container':
        return 3
    if label == 'Entity-Destination':
        return 4
    if label == 'Entity-Origin':
        return 5
    if label == 'Instrument-Agency':
        return 6
    if label == 'Member-Collection':
        return 7
    if label == 'Message-Topic':
        return 8
    else:
        return 9


def generate_range(start, stop):
    result = []
    for i in range(start, stop):
        result.append(i)
    return result

def getCorpus(from_file,to_file):
    dic = {}
    types = set()
    save_file = open(to_file, "w")
    with open(from_file, 'r',
              encoding='utf-8') as file:
        lines = file.readlines()
        state = 0
        for line in lines:
            line = line.strip()
            if state == 0:
                dic["index"] = line[0:line.find("\t")]
                sentence = line[line.find("\t") + 1:]
                word_index = 0
                for word in sentence.split(" "):
                    if word.__contains__('<e1>'):
                        dic["e1"] = str(word_index)
                    elif word.__contains__('<e2>'):
                        dic["e2"] = str(word_index)
                    word_index+=1
                dic["sentence"] = sentence.replace("<e1>", "").replace("<e2>", "").replace("</e1>", "").replace("</e2>","")[1:-1]
                state = 1
            elif state == 1:
                dic["type"] = line
                state = 2
            elif state == 2:
                state = 3
            elif state == 3:
                state = 0
                types.add(dic["type"])
                save_file.write(dic["type"])
                save_file.write("|")
                save_file.write(dic["e1"])
                save_file.write("|")
                save_file.write(dic["e2"])
                save_file.write("|")
                save_file.write(dic["sentence"])
                save_file.write("\n")
    print(types)
    save_file.close()


def getCorpusWithoutPosi(from_file,to_file):
    dic = {}
    types = set()
    save_file = open(to_file, "w")
    with open(from_file, 'r',
              encoding='utf-8') as file:
        lines = file.readlines()
        state = 0
        for line in lines:
            line = line.strip()
            if state == 0:
                dic["index"] = line[0:line.find("\t")]
                sentence = line[line.find("\t") + 1:]
                word_index = 0
                for word in sentence.split(" "):
                    if word.__contains__('<e1>'):
                        dic["e1"] = str(word_index)
                    elif word.__contains__('<e2>'):
                        dic["e2"] = str(word_index)
                    word_index+=1
                dic["sentence"] = sentence[1:-1]
                state = 1
            elif state == 1:
                dic["type"] = line
                state = 2
            elif state == 2:
                state = 3
            elif state == 3:
                state = 0
                types.add(dic["type"])
                save_file.write(dic["type"])
                save_file.write("|")
                save_file.write(dic["sentence"])
                save_file.write("\n")
    print(types)
    save_file.close()


if __name__ == '__main__':
    train_from_file = "../data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    test_from_file = "../data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
    train_to_file = "../data/train_en.txt"
    test_to_file = "../data/test_en.txt"
    getCorpusWithoutPosi(train_from_file,train_to_file)
    getCorpusWithoutPosi(test_from_file,test_to_file)
