# src/kffl/app/message_types.py

# FAIR1: clients compute local fairness/RFF terms and send to server
QUERY_FAIR1 = "query.fair1"

# KFFL train step: server sends (model + fairness constraint) and clients do local update
TRAIN_KFFL = "train.kffl"
