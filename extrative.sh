# for input data
python3.6 ./src/preprocess_seq_tag.py ./datasets/seq_tag/ $1 
# for test
python3.6 ./src/extractive_test.py ./datasets/seq_tag/test_tag.pkl $2 
