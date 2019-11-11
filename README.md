# NLP at SemEval-2019 Task6: Detecting Offensive language using Neural Networks
This is a repository containing models submitted as part of our participation in SemEval Task 6 OffensEval: Idenstifying and categorizing Offensive lnaguage in social media by SemEval 2019 by (Zampieri et al.,2019b)

# Files
1. Preprocess.py Some of basic Preprocessing as Step 1 to clean the data.
2. TaskA.py: It is a classification model dealing with 2 classes.
3. TaskB.py: This model is utilizing both the word and pos features. At first POS Tag for all the sentences can be get by     POS_tag.py in form of text file and then inserted at line 50. 
4. TaskC.py : This is for the third task i.e Target Identification
5. POS_tag.py: This can be used to get the POS tag for all the sentences using nltk.


# Dependencies
Python:2.7.5 &#12288;
numpy: 1.16.0 &#12288;
gensim: 3.7.0 &#12288;
tensorflow: 1.14.0 &#12288;
nltk:3.4 &#12288;
pandas: 0.23.4 &#12288;






# References
@inproceedings{kapil2019nlp,
  title={NLP at SemEval-2019 Task 6: Detecting Offensive language using Neural Networks},
  author={Kapil, Prashant and Ekbal, Asif and Das, Dipankar},
  booktitle={Proceedings of the 13th International Workshop on Semantic Evaluation},
  pages={587--592},
  year={2019}
}
