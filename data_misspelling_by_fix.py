# Misspelling Dim Articles CSV by rate

# Library

# Data Processing
import numpy as np
import pandas as pd
import re
import data_cleaning as dc
import random


# Load Dataset
true_articles = pd.read_csv('resources/train_dataset/dim_articles.csv', header=None)


# Misspelling
#0 @#$
#1 i/y
# im -> ym,  ai -> ay,  ye^ -> ie^
#2 l/n
# l -> n
#3 ch/tr
# ch -> tr,  tr -> ch,  tr -> gi
#4 s/x
# s -> x,  x -> s
#5 r/d/gi
# d <-> gi,  d <-> r,  r <-> gi,  đ <-> d
#6 c/k
# c -> k,  k -> c
#7 o/ô/ơ/u/ư
# o <-> ô <-> ơ <-> u <-> ư
#8 a/e
# a <-> e
#9 n/nh/ng
# n <-> nh <-> ng <-> ngh
#10 ao/au
# ao <-> au
#11 '/~/?
# ' <-> ~ <-> ?
#12 at/ac, ut/uc, ot/oc, iet/iec
# at <-> ac
#13 ph/v
# ph <-> v
#14 swap
#15 dup

#1 i/y
def misspell_1(text):
    ms_set = set()
    ms_set.add(text)
    # i -> y
    ms_set.add(re.sub(r"I", "Y", text))
    ms_set.add(re.sub(r"i", "y", text))
    ms_set.add(re.sub(r"í", "ý", text))
    ms_set.add(re.sub(r"ì", "ỳ", text))
    ms_set.add(re.sub(r"ị", "ỵ", text))
    ms_set.add(re.sub(r"ĩ", "ỹ", text))
    ms_set.add(re.sub(r"ỉ", "ỷ", text))
    # y -> i
    ms_set.add(re.sub(r"Y", "I", text))
    ms_set.add(re.sub(r"y", "i", text))
    ms_set.add(re.sub(r"ý", "í", text))
    ms_set.add(re.sub(r"ỳ", "ì", text))
    ms_set.add(re.sub(r"ỵ", "ị", text))
    ms_set.add(re.sub(r"ỹ", "ĩ", text))
    ms_set.add(re.sub(r"ỷ", "ỉ", text))
    ms_set.remove(text)
    return ms_set

#2 l/n
def misspell_2(text):
    ms_set = set()
    ms_set.add(text)
    # l -> n
    ms_set.add(re.sub(r"L", "N", text))
    ms_set.add(re.sub(r"l", "n", text))
    # _n -> l
    ms_set.add(re.sub(r"N(?![gh])", "L", text))
    ms_set.add(re.sub(r" n(?![gh])", " l", text))
    ms_set.remove(text)
    return ms_set

#3 ch/tr
def misspell_3(text):
    ms_set = set()
    ms_set.add(text)
    # _ch -> tr
    ms_set.add(re.sub(r"Ch", "Tr", text))
    ms_set.add(re.sub(r" ch", " tr", text))
    # tr -> ch
    ms_set.add(re.sub(r"Tr", "Ch", text))
    ms_set.add(re.sub(r"tr", "ch", text))
    # tr -> gi
    ms_set.add(re.sub(r"Tr", "Gi", text))
    ms_set.add(re.sub(r"tr", "gi", text))
    ms_set.remove(text)
    return ms_set

#4 s/x
def misspell_4(text):
    ms_set = set()
    ms_set.add(text)
    # s -> x
    ms_set.add(re.sub(r"S", "X", text))
    ms_set.add(re.sub(r"s", "x", text))
    # x -> s
    ms_set.add(re.sub(r"X", "S", text))
    ms_set.add(re.sub(r"x", "s", text))
    ms_set.remove(text)
    return ms_set

#5 r/d/gi
def misspell_5(text):
    ms_set = set()
    ms_set.add(text)
    # d <-> gi
    ms_set.add(re.sub(r"D", "Gi", text))
    ms_set.add(re.sub(r"d", "gi", text))
    ms_set.add(re.sub(r"Gi", "D", text))
    ms_set.add(re.sub(r"gi", "d", text))    
    # d <-> r
    ms_set.add(re.sub(r"D", "R", text))
    ms_set.add(re.sub(r"d", "r", text))
    ms_set.add(re.sub(r"R", "D", text))
    ms_set.add(re.sub(r" r", " d", text))
    # r <-> gi
    ms_set.add(re.sub(r"R", "Gi", text))
    ms_set.add(re.sub(r" r", " gi", text))
    ms_set.add(re.sub(r"Gi", "R", text))
    ms_set.add(re.sub(r"gi", "r", text))
    # đ <-> d
    ms_set.add(re.sub(r"Đ", "D", text))
    ms_set.add(re.sub(r"đ", "d", text))
    ms_set.add(re.sub(r"D", "Đ", text))
    ms_set.add(re.sub(r"d", "đ", text))
    ms_set.remove(text)
    return ms_set

#6 c/k
def misspell_6(text):
    ms_set = set()
    ms_set.add(text)
    # c -> k
    ms_set.add(re.sub(r"C", "K", text))
    ms_set.add(re.sub(r" c", " k", text))
    # k -> c
    ms_set.add(re.sub(r"K", "C", text))
    ms_set.add(re.sub(r"k", "c", text))
    ms_set.remove(text)
    return ms_set

#7 o/ô/ơ/u/ư
def misspell_7(text):
    ms_set = set()
    ms_set.add(text)
    # o -> ô ơ u ư
    ms_set.add(re.sub(r"O", "Ô", text))
    ms_set.add(re.sub(r"O", "Ơ", text))
    ms_set.add(re.sub(r"O", "U", text))
    ms_set.add(re.sub(r"O", "Ư", text))
    ms_set.add(re.sub(r"o", "ô", text))
    ms_set.add(re.sub(r"ó", "ố", text))
    ms_set.add(re.sub(r"ò", "ồ", text))
    ms_set.add(re.sub(r"ọ", "ộ", text))
    ms_set.add(re.sub(r"õ", "ỗ", text))
    ms_set.add(re.sub(r"ỏ", "ổ", text))
    ms_set.add(re.sub(r"o", "ơ", text))
    ms_set.add(re.sub(r"ó", "ớ", text))
    ms_set.add(re.sub(r"ò", "ờ", text))
    ms_set.add(re.sub(r"ọ", "ợ", text))
    ms_set.add(re.sub(r"õ", "ỡ", text))
    ms_set.add(re.sub(r"ỏ", "ở", text))
    ms_set.add(re.sub(r"o", "u", text))
    ms_set.add(re.sub(r"ó", "ú", text))
    ms_set.add(re.sub(r"ò", "ù", text))
    ms_set.add(re.sub(r"ọ", "ụ", text))
    ms_set.add(re.sub(r"õ", "ũ", text))   
    ms_set.add(re.sub(r"ỏ", "ủ", text))
    ms_set.add(re.sub(r"o", "ư", text))
    ms_set.add(re.sub(r"ó", "ứ", text))
    ms_set.add(re.sub(r"ò", "ừ", text))
    ms_set.add(re.sub(r"ọ", "ự", text))
    ms_set.add(re.sub(r"õ", "ữ", text))
    ms_set.add(re.sub(r"ỏ", "ử", text))
    # ô -> o ơ u ư
    ms_set.add(re.sub(r"Ô", "O", text))
    ms_set.add(re.sub(r"Ô", "Ơ", text))
    ms_set.add(re.sub(r"Ô", "U", text))
    ms_set.add(re.sub(r"Ô", "Ư", text))
    ms_set.add(re.sub(r"ô", "o", text))
    ms_set.add(re.sub(r"ố", "ó", text))
    ms_set.add(re.sub(r"ồ", "ò", text))
    ms_set.add(re.sub(r"ộ", "ọ", text))
    ms_set.add(re.sub(r"ỗ", "õ", text))
    ms_set.add(re.sub(r"ổ", "ỏ", text))
    ms_set.add(re.sub(r"ô", "ơ", text))
    ms_set.add(re.sub(r"ố", "ớ", text))
    ms_set.add(re.sub(r"ồ", "ờ", text))
    ms_set.add(re.sub(r"ộ", "ợ", text))
    ms_set.add(re.sub(r"ỗ", "ỡ", text))
    ms_set.add(re.sub(r"ổ", "ở", text))
    ms_set.add(re.sub(r"ô", "u", text))
    ms_set.add(re.sub(r"ố", "ú", text))
    ms_set.add(re.sub(r"ồ", "ù", text))
    ms_set.add(re.sub(r"ộ", "ụ", text))
    ms_set.add(re.sub(r"ỗ", "ũ", text))
    ms_set.add(re.sub(r"ổ", "ủ", text))
    ms_set.add(re.sub(r"ô", "ư", text))
    ms_set.add(re.sub(r"ố", "ứ", text))
    ms_set.add(re.sub(r"ồ", "ừ", text))
    ms_set.add(re.sub(r"ộ", "ự", text))
    ms_set.add(re.sub(r"ỗ", "ữ", text))
    ms_set.add(re.sub(r"ổ", "ử", text))
    # ơ -> o ô u ư
    ms_set.add(re.sub(r"Ơ", "O", text))
    ms_set.add(re.sub(r"Ơ", "Ô", text))
    ms_set.add(re.sub(r"Ơ", "U", text))
    ms_set.add(re.sub(r"Ơ", "Ư", text))
    ms_set.add(re.sub(r"ơ", "o", text))
    ms_set.add(re.sub(r"ớ", "ó", text))
    ms_set.add(re.sub(r"ờ", "ò", text))
    ms_set.add(re.sub(r"ợ", "ọ", text))
    ms_set.add(re.sub(r"ỡ", "õ", text))
    ms_set.add(re.sub(r"ở", "ỏ", text))
    ms_set.add(re.sub(r"ơ", "ô", text))
    ms_set.add(re.sub(r"ớ", "ố", text))
    ms_set.add(re.sub(r"ờ", "ồ", text))
    ms_set.add(re.sub(r"ợ", "ộ", text))
    ms_set.add(re.sub(r"ỡ", "ỗ", text))
    ms_set.add(re.sub(r"ở", "ổ", text))
    ms_set.add(re.sub(r"ơ", "u", text))
    ms_set.add(re.sub(r"ớ", "ú", text))
    ms_set.add(re.sub(r"ờ", "ù", text))
    ms_set.add(re.sub(r"ợ", "ụ", text))
    ms_set.add(re.sub(r"ỡ", "ũ", text))
    ms_set.add(re.sub(r"ở", "ủ", text))
    ms_set.add(re.sub(r"ơ", "ư", text))
    ms_set.add(re.sub(r"ớ", "ứ", text))
    ms_set.add(re.sub(r"ờ", "ừ", text))
    ms_set.add(re.sub(r"ợ", "ự", text))
    ms_set.add(re.sub(r"ỡ", "ữ", text))
    ms_set.add(re.sub(r"ở", "ử", text))
    # u -> o ô ơ ư
    ms_set.add(re.sub(r"U", "O", text))
    ms_set.add(re.sub(r"U", "Ô", text))
    ms_set.add(re.sub(r"U", "Ơ", text))
    ms_set.add(re.sub(r"U", "Ư", text))
    ms_set.add(re.sub(r"u", "o", text))
    ms_set.add(re.sub(r"ú", "ó", text))
    ms_set.add(re.sub(r"ù", "ò", text))
    ms_set.add(re.sub(r"ụ", "ọ", text))
    ms_set.add(re.sub(r"ũ", "õ", text))
    ms_set.add(re.sub(r"ủ", "ỏ", text))
    ms_set.add(re.sub(r"u", "ô", text))
    ms_set.add(re.sub(r"ú", "ố", text))
    ms_set.add(re.sub(r"ù", "ồ", text))
    ms_set.add(re.sub(r"ụ", "ộ", text))
    ms_set.add(re.sub(r"ũ", "ỗ", text))
    ms_set.add(re.sub(r"ủ", "ổ", text))
    ms_set.add(re.sub(r"u", "ơ", text))
    ms_set.add(re.sub(r"ú", "ớ", text))
    ms_set.add(re.sub(r"ù", "ờ", text))
    ms_set.add(re.sub(r"ụ", "ợ", text))
    ms_set.add(re.sub(r"ũ", "ỡ", text))
    ms_set.add(re.sub(r"ủ", "ở", text))
    ms_set.add(re.sub(r"u", "ư", text))
    ms_set.add(re.sub(r"ú", "ứ", text))
    ms_set.add(re.sub(r"ù", "ừ", text))
    ms_set.add(re.sub(r"ụ", "ự", text))
    ms_set.add(re.sub(r"ũ", "ữ", text))
    ms_set.add(re.sub(r"ủ", "ử", text))
    # ư -> o ô ơ u
    ms_set.add(re.sub(r"Ư", "O", text))
    ms_set.add(re.sub(r"Ư", "Ô", text))
    ms_set.add(re.sub(r"Ư", "Ơ", text))
    ms_set.add(re.sub(r"Ư", "U", text))
    ms_set.add(re.sub(r"ư", "o", text))
    ms_set.add(re.sub(r"ứ", "ó", text))
    ms_set.add(re.sub(r"ừ", "ò", text))
    ms_set.add(re.sub(r"ự", "ọ", text))
    ms_set.add(re.sub(r"ữ", "õ", text))
    ms_set.add(re.sub(r"ử", "ỏ", text))
    ms_set.add(re.sub(r"ư", "ô", text))
    ms_set.add(re.sub(r"ứ", "ố", text))
    ms_set.add(re.sub(r"ừ", "ồ", text))
    ms_set.add(re.sub(r"ự", "ộ", text))
    ms_set.add(re.sub(r"ữ", "ỗ", text))
    ms_set.add(re.sub(r"ử", "ổ", text))
    ms_set.add(re.sub(r"ư", "ơ", text))
    ms_set.add(re.sub(r"ứ", "ớ", text))
    ms_set.add(re.sub(r"ừ", "ờ", text))
    ms_set.add(re.sub(r"ự", "ợ", text))
    ms_set.add(re.sub(r"ữ", "ỡ", text))
    ms_set.add(re.sub(r"ử", "ở", text))
    ms_set.add(re.sub(r"ư", "u", text))
    ms_set.add(re.sub(r"ứ", "ú", text))
    ms_set.add(re.sub(r"ừ", "ù", text))
    ms_set.add(re.sub(r"ự", "ụ", text))
    ms_set.add(re.sub(r"ữ", "ũ", text))
    ms_set.add(re.sub(r"ử", "ủ", text))
    ms_set.remove(text)
    return ms_set
      
#8 a/e
def misspell_8(text):
    ms_set = set()
    ms_set.add(text)
    # a -> e
    ms_set.add(re.sub(r"A", "E", text))
    ms_set.add(re.sub(r"Á", "É", text))
    ms_set.add(re.sub(r"a", "e", text))
    ms_set.add(re.sub(r"á", "é", text))
    ms_set.add(re.sub(r"à", "è", text))
    ms_set.add(re.sub(r"ạ", "ẹ", text))
    ms_set.add(re.sub(r"ã", "ẽ", text))
    ms_set.add(re.sub(r"ả", "ẻ", text))
    # e -> a
    ms_set.add(re.sub(r"E", "A", text))
    ms_set.add(re.sub(r"É", "Á", text))
    ms_set.add(re.sub(r"e", "a", text))
    ms_set.add(re.sub(r"é", "á", text))
    ms_set.add(re.sub(r"è", "à", text))
    ms_set.add(re.sub(r"ẹ", "ạ", text))
    ms_set.add(re.sub(r"ẽ", "ã", text))
    ms_set.add(re.sub(r"ẻ", "ả", text))
    ms_set.remove(text)
    return ms_set
    
#9 n/nh/ng
def misspell_9(text):
    ms_set = set()
    ms_set.add(text)
    # n -> nh ng
    ms_set.add(re.sub(r"N(?![gh])", "Nh", text))
    ms_set.add(re.sub(r"N(?![gh])", "Ng", text))
    ms_set.add(re.sub(r"n(?![gh])", "nh", text))
    ms_set.add(re.sub(r"n(?![gh])", "ng", text))
    # nh -> n ng
    ms_set.add(re.sub(r"Nh", "N", text))
    ms_set.add(re.sub(r"Nh", "Ng", text))
    ms_set.add(re.sub(r"nh", "n", text))
    ms_set.add(re.sub(r"nh", "ng", text))
    # ng -> n nh ngh
    ms_set.add(re.sub(r"Ng(?!h)", "N", text))
    ms_set.add(re.sub(r"Ng(?!h)", "Nh", text))
    ms_set.add(re.sub(r"Ng(?!h)", "Ngh", text))
    ms_set.add(re.sub(r"ng(?!h)", "n", text))
    ms_set.add(re.sub(r"ng(?!h)", "nh", text))
    ms_set.add(re.sub(r"ng(?!h)", "ngh", text))
    ms_set.remove(text)
    return ms_set

#10 ao/au/âu
def misspell_10(text):
    ms_set = set()
    ms_set.add(text)
    # ao -> au âu
    ms_set.add(re.sub(r"ao", "au", text))
    ms_set.add(re.sub(r"áo", "áu", text))
    ms_set.add(re.sub(r"ào", "àu", text))
    ms_set.add(re.sub(r"ạo", "ạu", text))
    ms_set.add(re.sub(r"ão", "ãu", text))
    ms_set.add(re.sub(r"ảo", "ảu", text))
    ms_set.add(re.sub(r"ao", "âu", text))
    ms_set.add(re.sub(r"áo", "ấu", text))
    ms_set.add(re.sub(r"ào", "ầu", text))
    ms_set.add(re.sub(r"ạo", "ậu", text))
    ms_set.add(re.sub(r"ão", "ẫu", text))
    ms_set.add(re.sub(r"ảo", "ẩu", text))
    # au -> ao âu
    ms_set.add(re.sub(r"au", "ao", text))
    ms_set.add(re.sub(r"áu", "áo", text))
    ms_set.add(re.sub(r"àu", "ào", text))
    ms_set.add(re.sub(r"ạu", "ạo", text))
    ms_set.add(re.sub(r"ãu", "ão", text))
    ms_set.add(re.sub(r"ảu", "ảo", text))
    ms_set.add(re.sub(r"au", "âu", text))
    ms_set.add(re.sub(r"áu", "ấu", text))
    ms_set.add(re.sub(r"àu", "ầu", text))
    ms_set.add(re.sub(r"ạu", "ậu", text))
    ms_set.add(re.sub(r"ãu", "ẫu", text))
    ms_set.add(re.sub(r"ảu", "ẩu", text))
    # âu -> ao au
    ms_set.add(re.sub(r"âu", "ao", text))
    ms_set.add(re.sub(r"ấu", "áo", text))
    ms_set.add(re.sub(r"ầu", "ào", text))
    ms_set.add(re.sub(r"ậu", "ạo", text))
    ms_set.add(re.sub(r"ẫu", "ão", text))
    ms_set.add(re.sub(r"ẩu", "ảo", text))
    ms_set.add(re.sub(r"âu", "au", text))
    ms_set.add(re.sub(r"ấu", "áu", text))
    ms_set.add(re.sub(r"ầu", "àu", text))
    ms_set.add(re.sub(r"ậu", "ạu", text))
    ms_set.add(re.sub(r"ẫu", "ãu", text))
    ms_set.add(re.sub(r"ẩu", "ảu", text))
    ms_set.remove(text)
    return ms_set
    
#11 '/~/?
def misspell_11(text):
    ms_set = set()
    ms_set.add(text)
    # a â ă '~?
    ms_set.add(re.sub(r"á", "ã", text))
    ms_set.add(re.sub(r"á", "ả", text))
    ms_set.add(re.sub(r"ã", "á", text))
    ms_set.add(re.sub(r"ã", "ả", text))
    ms_set.add(re.sub(r"ả", "á", text))
    ms_set.add(re.sub(r"ả", "ã", text))
    ms_set.add(re.sub(r"ấ", "ẫ", text))
    ms_set.add(re.sub(r"ấ", "ẩ", text))
    ms_set.add(re.sub(r"ẫ", "ấ", text))
    ms_set.add(re.sub(r"ẫ", "ẩ", text))
    ms_set.add(re.sub(r"ẩ", "ấ", text))
    ms_set.add(re.sub(r"ẩ", "ẫ", text))
    ms_set.add(re.sub(r"ắ", "ẵ", text))
    ms_set.add(re.sub(r"ắ", "ẳ", text))
    ms_set.add(re.sub(r"ẵ", "ắ", text))
    ms_set.add(re.sub(r"ẵ", "ẳ", text))
    ms_set.add(re.sub(r"ẳ", "ắ", text))
    ms_set.add(re.sub(r"ẳ", "ẵ", text))
    # e ê '~?
    ms_set.add(re.sub(r"é", "ẽ", text))
    ms_set.add(re.sub(r"é", "ẻ", text))
    ms_set.add(re.sub(r"ẽ", "é", text))
    ms_set.add(re.sub(r"ẽ", "ẻ", text))
    ms_set.add(re.sub(r"ẻ", "é", text))
    ms_set.add(re.sub(r"ẻ", "ẽ", text))
    ms_set.add(re.sub(r"ế", "ễ", text))
    ms_set.add(re.sub(r"ế", "ể", text))
    ms_set.add(re.sub(r"ễ", "ế", text))
    ms_set.add(re.sub(r"ễ", "ể", text))
    ms_set.add(re.sub(r"ể", "ế", text))
    ms_set.add(re.sub(r"ể", "ễ", text))
    # o ô ơ '~?
    ms_set.add(re.sub(r"ó", "õ", text))
    ms_set.add(re.sub(r"ó", "ỏ", text))
    ms_set.add(re.sub(r"õ", "ó", text))
    ms_set.add(re.sub(r"õ", "ỏ", text))
    ms_set.add(re.sub(r"ỏ", "ó", text))
    ms_set.add(re.sub(r"ỏ", "õ", text))
    ms_set.add(re.sub(r"ố", "ỗ", text))
    ms_set.add(re.sub(r"ố", "ổ", text))
    ms_set.add(re.sub(r"ỗ", "ố", text))
    ms_set.add(re.sub(r"ỗ", "ổ", text))
    ms_set.add(re.sub(r"ổ", "ố", text))
    ms_set.add(re.sub(r"ổ", "ỗ", text))
    ms_set.add(re.sub(r"ớ", "ỡ", text))
    ms_set.add(re.sub(r"ớ", "ở", text))
    ms_set.add(re.sub(r"ỡ", "ớ", text))
    ms_set.add(re.sub(r"ỡ", "ở", text))
    ms_set.add(re.sub(r"ở", "ớ", text))
    ms_set.add(re.sub(r"ở", "ỡ", text))
    # u ư '~?
    ms_set.add(re.sub(r"ú", "ũ", text))
    ms_set.add(re.sub(r"ú", "ủ", text))
    ms_set.add(re.sub(r"ũ", "ú", text))
    ms_set.add(re.sub(r"ũ", "ủ", text))
    ms_set.add(re.sub(r"ủ", "ú", text))
    ms_set.add(re.sub(r"ủ", "ũ", text))
    ms_set.add(re.sub(r"ứ", "ữ", text))
    ms_set.add(re.sub(r"ứ", "ử", text))
    ms_set.add(re.sub(r"ữ", "ứ", text))
    ms_set.add(re.sub(r"ữ", "ử", text))
    ms_set.add(re.sub(r"ử", "ứ", text))
    ms_set.add(re.sub(r"ử", "ữ", text))
    # i y '~?
    ms_set.add(re.sub(r"í", "ĩ", text))
    ms_set.add(re.sub(r"í", "ỉ", text))
    ms_set.add(re.sub(r"ĩ", "í", text))
    ms_set.add(re.sub(r"ĩ", "ỉ", text))
    ms_set.add(re.sub(r"ỉ", "í", text))
    ms_set.add(re.sub(r"ỉ", "ĩ", text))
    ms_set.add(re.sub(r"ý", "ỹ", text))
    ms_set.add(re.sub(r"ý", "ỷ", text))
    ms_set.add(re.sub(r"ỹ", "ý", text))
    ms_set.add(re.sub(r"ỹ", "ỷ", text))
    ms_set.add(re.sub(r"ỷ", "ý", text))
    ms_set.add(re.sub(r"ỷ", "ỹ", text))
    ms_set.remove(text)
    return ms_set
     
#12 at/ac, ut/uc, ot/oc, iet/iec
def misspell_12(text):
    ms_set = set()
    ms_set.add(text)  
    # *t <-> *c
    ms_set.add(re.sub(r"(?<! )c(?!h)", "t", text))
    ms_set.add(re.sub(r"(?<! )t", "c", text))
    ms_set.remove(text)
    return ms_set

#13 ph/v
def misspell_13(text):
    ms_set = set()
    ms_set.add(text)  
    # ph <-> v
    ms_set.add(re.sub(r"Ph", "V", text))
    ms_set.add(re.sub(r"ph", "v", text))
    ms_set.add(re.sub(r"V", "Ph", text))
    ms_set.add(re.sub(r"v", "ph", text))
    ms_set.remove(text)
    return ms_set

#14 <-> swap
def misspell_14(text):
    ms_set = set()
    ms_set.add(text)
    # <->
    #random.seed(369)
    split_text = text.split(" ")
    if len(split_text) < 2:
        return ms_set
    
    for i in range(2):
        indices = random.sample(range(len(split_text)), 2)
        first = min(indices)
        second = max(indices)
        mstext = " ".join([x for x in split_text[:first]]) + " " + split_text[second] + " "
        mstext = mstext + " ".join([x for x in split_text[(first+1):second]]) + " " + split_text[first] + " "
        mstext = mstext + " ".join([x for x in split_text[(second+1):]])
        ms_set.add(mstext)
    ms_set.remove(text)
    return ms_set
    
#15 <> dup
def misspell_15(text):
    ms_set = set()
    ms_set.add(text)
    # <>
    split_text = text.split(" ")
    for i in range(2):
        index = random.sample(range(len(split_text)), 1)
        index = min(index)
        mstext = " ".join([x for x in split_text[:index]]) + " " + split_text[index] + " " + split_text[index] + " "
        mstext = mstext + " ".join([x for x in split_text[(index+1):]])
        ms_set.add(mstext)
    ms_set.remove(text)
    return ms_set


#[i]
def misspell_i(text, i):
    if i <= 0:
        return misspell_1(text)
    if i == 1:
        return misspell_2(text)
    if i == 2:
        return misspell_3(text)
    if i == 3:
        return misspell_4(text)
    if i == 4:
        return misspell_5(text)
    if i == 5:
        return misspell_6(text)
    if i == 6:
        return misspell_7(text)
    if i == 7:
        return misspell_8(text)
    if i == 8:
        return misspell_9(text)
    if i == 9:
        return misspell_10(text)
    if i == 10:
        return misspell_11(text)
    if i == 11:
        return misspell_12(text)
    if i == 12:
        return misspell_13(text)
    if i == 13:
        return misspell_14(text)
    if i >= 14:
        return misspell_15(text)
    

# Compute rate of misspell text
def get_rate(text, stext, i):
    _arr1 = text.split(" ")
    _arr2 = stext.split(" ")
    _set1 = set(_arr1)
    _set2 = set(_arr2)
    _set1.update(_set2)
    
    if i == 13: # swap
        rate = 2.0 / len(_set2)
        return rate
    if i >= 14: # dup
        rate = 1.0 / len(_arr2)
        return rate 
    rate = (len(_set1) - len(_set2)) / len(_set2)
    return rate
    

# Fragment true articles
dim_articles = true_articles
dim_articles = dim_articles.sample(frac=1, random_state=369).reset_index(drop=True)
fragments = []
for i in range(14):
    sampling = dim_articles.sample(n=26271, random_state=369)
    dim_articles = dim_articles.drop(sampling.index)
    fragments.append(sampling)
    if i == 13:
        fragments.append(dim_articles)


# Generate misspelling sets    
ms_set = []
ms_output = []
for i in range(15):
    for text in fragments[i].values:
        mstext = misspell_i(str(text),i)
        for stext in mstext:
            ms_set.append(str(stext))
            ms_output.append(str(text))
        
    

# Clean texts, Sampling misspell sets to make false articles df
false_input_df = []
false_output_df = []
for text in ms_set:
    false_input_df.append(dc.clean_paragraphs(str(text)))
    
for text in ms_output:
    false_output_df.append(dc.clean_paragraphs(str(text)))


false_input_articles = pd.DataFrame(false_input_df)
false_output_articles = pd.DataFrame(false_output_df)


        
        
# Save true and false articles into csv
false_input_articles = false_input_articles.sample(frac=1, random_state=369).reset_index(drop=True)
false_output_articles = false_output_articles.sample(frac=1, random_state=369).reset_index(drop=True)
#false_articles.to_csv('resources/train_dataset/false_articles.csv', index=False, header=False)
false_input_articles.to_csv('resources/train_dataset/false_input_articles_full.csv', index=False, header=False)
false_output_articles.to_csv('resources/train_dataset/false_output_articles_full.csv', index=False, header=False)
#true_articles.to_csv('resources/train_dataset/true_articles.csv', index=False, header=False)
    
