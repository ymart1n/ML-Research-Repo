# install.packages('jiebaR')
# install.packages("readr")
library(readr)
library(jiebaR)
test_string <- read_file("./Homework 5/test.txt")
# mixseg <- worker()
# segment( "这是一段测试文本" , mixseg ) 
#或者用以下操作
# mixseg['这是一段测试文本']
mixseg <= test_string

# words = "我爱北京天安门"
tagger = worker("tag") #开启词性标注启发器
tagger <= test_string

keys = worker("keywords",topn = 5, idf = IDFPATH)
keys <= test_string