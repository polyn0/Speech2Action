import os
import json
import nltk
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
import itertools
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import word_tokenize, pos_tag, ne_chunk, FreqDist

path_dir = './ParserOutput/Action'
file_list = os.listdir(path_dir)
# print(file_list)

stage_direction_list = []
speech_list = []


def stage_direction_to_verb(stage_directions):
    verb_list = []
    # idx = 0
    # stage direction 문장 하나씩 접근
    for i in tqdm(stage_directions):
        # if idx==5:
        #     print(verb_list[:5])
        #     break
        sentence = pos_tag(word_tokenize(i))
        # print(idx, sentence)
        # idx+=1 #제거대상
        verb = []
        for word, tag in sentence:
            if tag in ['VB', 'VBN', 'VBD', 'VBG', 'VBP', 'VBZ']:
                verb.append(word.lower())
        verb_list.append(verb)
    print(len(verb_list), verb_list[0])

    # # 혹시몰라 확인, 전부 None처리돼 들어감. action verb 없는 stage direction 존재함.
    # if len(stage_directions) != len(verb_list):
    #     print(len(stage_directions), len(verb_list))
    return verb_list


# 빈도수를 세려면 어차피 모든 단어가 다 있어야함.
# for문 안에서 하는건... 차라리 그냥 따로 빼는게 나을듯
def verb_preprocess(verbs):
    print('확인(문장개수)', len(verbs))

    # vocab[v1,2,3,4,....] 문장별로 분리되어있던 것 하나로.
    vocab = np.hstack(verbs)
    print('잘 펴졌나 확인', vocab.shape, 'vocab 수 총합!!!!!')


    # 2. vocab 빈도수로 변환 -> vocab_freq[(word, freq)]
    vocab_freq = FreqDist(vocab)

    # 3. 빈도수 높은거 100개, 50보다 작은거 제거 -> ""통과한 거"" stopwords에 저장.
    # 빈도수 순으로 나열
    vocab_freq = vocab_freq.most_common()

    # 3-1) 빈도수 높은 100개 제거   # vocab = list(vocab.items())[100:]
    vocab_freq = vocab_freq[100:]
    print(len(vocab_freq))

    # 3-2) 빈도수 50 이하 제거 -> vocab_freq2
    vocab_freq2 = {}
    for (word, frequency) in vocab_freq:
        if frequency > 10:  # 빈도수가 작은 단어는 제외. --> 범위 넓혔을 때 50으로 바꿔줘야!!
            # 어차피 word 다 다르니 overwrite 될 일 X
            vocab_freq2[word] = frequency
    print(list(vocab_freq2.items())[-5:]) # 끝에 5단어 관찰

    # 3-3) 통과할 verb들을 stopwords로 만듦.
    print('비교!!! 100개도 아니고 50 이상 단어들.',len(vocab_freq2))
    stopwords = [x for x in vocab_freq2.keys()]

    # 4. verbs([문장][v1, v2, ...])에서 stopwords만 통과시킴.
    for i in tqdm(range(len(verbs))):
        verbs[i] = [w for w in verbs[i] if w in stopwords]

    print('확인(문장개수)', len(verbs), '위에꺼랑 똑같이 나오는지 확인')
    return verbs


def speech_preprocess(speechs) :
    for i in range(len(speechs)):
        speechs[i] = re.sub("[^a-zA-Z0-9\s]","", speechs[i])
    return speechs


def screenplay_parsing():
    # total_verb = []

    for i in file_list:
        if i.endswith('.pkl'):
            continue
        with open(path_dir + '/' + i, 'r') as f:
            dic = json.load(f)
            # print(len(dic))
            print('Opening!', i)

        # 대본.json 들어가서 EXT/INT 로 나뉜 거 하나씩 접근
        for j in range(len(dic)):
            for k in range(len(dic[j])):
                # stage direction 직후에 나오는 speech 찾기
                if dic[j][k].get('head_type') == 'speaker/title' and dic[j][k - 1].get('head_type') != 'speaker/title':
                    stage_direction = dic[j][k - 1].get('text')
                    speech = dic[j][k].get('text')
                    # print("Stage Direction: ".ljust(20), stage_direction)
                    # print("Speech:   ".ljust(20), speech, '\n')

                    # 예외처리 - speech 바로 전 stage direction text = None 방지
                    if stage_direction == "":
                        stage_direction = dic[j][k - 1].get('head_text').get('subj')
                        # print('None modified')

                    # 처리했는데도 stage_direction==None인거 list에서 제외 - None이 생각보다 많음
                    if stage_direction == None:
                        continue

                    stage_direction_list.append(stage_direction)
                    speech_list.append(speech)

    # 장르 1개(대본 n개)에 대한 stage direction - speech set
    print(len(stage_direction_list), len(speech_list))
    return stage_direction_list, speech_list
    # stage_direction_list, speech_list 저장.

    #     # total_verb [대본][stagedirection문장][verb]
    #     total_verb.append(stage_direction_to_verb(stage_direction_list))
    #
    #     break
    # print('대본개수', len(total_verb), len(file_list) / 2)
    # verb_preprocess(total_verb)



speech_list, stage_direction_list = screenplay_parsing()

speech_list = speech_preprocess(speech_list)

S2S = pd.DataFrame({'speech': speech_list,
                    'stage_direction': stage_direction_list})
print(S2S.head)
print(S2S.shape)

stage_direction_list = stage_direction_to_verb(stage_direction_list)
action_list = verb_preprocess(stage_direction_list)

# Speech2Action training data 구조로 구성
# 1. 결측값제거 https://rfriend.tistory.com/263
# 2. 행 분리? ㅠ늘려주기 http://daplus.net/python-%ED%8C%AC%EB%8D%94-%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%94%84%EB%A0%88%EC%9E%84-%EB%AC%B8%EC%9E%90%EC%97%B4-%ED%95%AD%EB%AA%A9%EC%9D%84-%EB%B6%84%ED%95%A0%ED%95%98%EC%97%AC-%ED%96%89-%EB%B6%84/
S2A = pd.DataFrame({'speech': action_list,
                    'stage_direction': stage_direction_list})

