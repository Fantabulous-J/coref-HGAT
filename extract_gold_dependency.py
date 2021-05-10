import json


for name in ['train', 'dev', 'test']:
    input_dir = "conll_data"
    input_path = "{}/{}.dependency.conllx".format(input_dir, name)
    output_dir = "conll_data"
    output_path = "{}/{}.english.depparse.gold.jsonlines".format(input_dir, name)

    documents = []
    document = {}
    sentence_features = []
    word_cnt = 0
    sent_word_cnt = 0
    with open(input_path) as input_file:
        for line in input_file.readlines():
            line = line.strip()
            if len(line) == 0:
                document['depparse_features'].append(sentence_features)
                sentence_features = []
                word_cnt += sent_word_cnt
                sent_word_cnt = 0
            elif not line[0].isdigit():
                documents.append(document)
                document = {'doc_key': line, 'depparse_features': []}
                word_cnt = 0
                sent_word_cnt = 0
            else:
                line = line.split('\t')
                id = int(line[0])
                word = line[1]
                head_id = int(line[6])
                dep_rel = line[7]
                sentence_features.append({"id": id + word_cnt,
                                          "word": word,
                                          "head_id": head_id + word_cnt if head_id != 0 else 0,
                                          "deprel": dep_rel})
                sent_word_cnt += 1
    documents.append(document)

    with open(output_path, "w") as output_file:
        for document in documents[1:]:
            output_file.write(json.dumps(document))
            output_file.write("\n")
