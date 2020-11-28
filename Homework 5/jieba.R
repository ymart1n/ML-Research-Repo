# install.packages('jiebaR')
# install.packages("readr")
# install.packages("pacman")
# install.packages("dplyr")
# install.packages("stringi", configure.vars="ICUDT_DIR=./icudt61l")
# install.packages("devtools")

library(pacman)
library(readr)
library(dplyr)
p_load(tidyverse,jiebaR)
p_load(tidytext)
library(jiebaR)

test_string <- read_file("./Homework 5/testML.txt")
# worker() -> wk
# segment(test_string, wk) -> mixseg

# mixseg

# tagger = worker("tag") #开启词性标注启发器
# vector_tag(mixseg, tagger) -> tag_result

# tag_result

# str(tag_result)

# tag_table <- enframe(tag_result)
# tag_table

keys = worker("keywords",topn = 100, idf = IDFPATH)
keywords <- keys <= test_string
str(keywords)

top_table <- data.frame(word=keywords, tf_idf=as.numeric(names(keywords)), row.names=NULL)

library(devtools)
devtools::install_github("lchiffon/wordcloud2")
library(wordcloud2)

wordcloud2(top_table, size = 2, fontFamily = "微软雅黑", color = "random-light", backgroundColor = "grey")
