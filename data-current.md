# Dataset Description
We are using [the Offensive Language Identification Dataset (OLID)](https://sites.google.com/site/offensevalsharedtask/olid), which contains a collection of 14,200 annotated English tweets using an annotation model. 

## Files
We splited the original train dataset into train dataset and development dataset:
- `train.csv`: 0.8
- `dev.csv`: 0.1
- `test.csv`: 0.1

## Columns
- `id`: Unique id that identifies a tweet
- `tweet`: Original tweets
- `subtask_a`: Label of a tweet. There are 2 categories: 
	- (NOT) Not Offensive - This post does not contain offense or profanity.
	- (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offense.

## An Example of the data

| id  | tweet | subtask_a 
| ------------- | ------------- | ------------- 
|86426|@USER She should ask a few native Americans what their take on this is. | OFF
|90194|@USER @USER Go home you’re drunk!!! @USER #MAGA #Trump2020 👊🇺🇸👊 URL|OFF
|16820|Amazon is investigating Chinese employees who are selling internal data to third-party sellers looking for an edge in the competitive marketplace. URL #Amazon #MAGA #KAG #CHINA #TCOT|NOT

## Dataset Reference
```
@inproceedings{zampierietal2019, 
title={{Predicting the Type and Target of Offensive Posts in Social Media}}, 
author={Zampieri, Marcos and Malmasi, Shervin and Nakov, Preslav and Rosenthal, Sara and Farra, Noura and Kumar, Ritesh}, 
booktitle={Proceedings of NAACL}, 
year={2019}, 
} 
```
