#!/usr/bin/env bash
pip install -r requirements.txt
python3.6 -m nltk.downloader all
python3.6 -m spacy download en_core_web_sm