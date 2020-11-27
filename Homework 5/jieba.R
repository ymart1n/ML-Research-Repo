# install.packages('jiebaR')
# install.packages("readr")
# install.packages("pacman")
# install.packages("dplyr")
# install.packages("stringi", configure.vars="ICUDT_DIR=./icudt61l")

library(pacman)
library(readr)
library(dplyr)
p_load(tidyverse,jiebaR)
p_load(tidytext)
library(jiebaR)

test_string <- read_file("./Homework 5/test.txt")
worker() -> wk
segment(test_string, wk) -> mixseg

# mixseg

tagger = worker("tag") #开启词性标注启发器
vector_tag(mixseg, tagger) -> tag_result

tag_result

str(tag_result)

tag_table <- enframe(tag_result)
tag_table

keys = worker("keywords",topn = 10, idf = IDFPATH)
keys <= test_string


p_load(wordcloud2)
tag_table %>% 
  top_n(10) %>% 
  wordcloud2(size = 2, fontFamily = "微软雅黑",
           color = "random-light", backgroundColor = "grey")
