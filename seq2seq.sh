# for input data
python3.6 ./src/preprocess_seq2seq.py ./datasets/seq2seq/ $1
# for test
python3.6 ./src/test.py ./datasets/seq2seq/test_seq.pkl $2 --isatt=0 --no=0 --load_model_path="./datasets/seq2seq/ckpt/model_0"
