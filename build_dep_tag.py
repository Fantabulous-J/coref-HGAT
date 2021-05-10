import json
from collections import Counter


def read_depparse_features(data_path):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    dependencies = []
    predicted_heads = []
    predicted_dependencies = []

    for example in examples:
        doc_key = example['doc_key']
        depparse_features = example['depparse_features']
        dependency_features = []
        head_features = []
        predicted_dependency_features = []
        for sent_feature in depparse_features:
            temp_dependency = []
            temp_predicted_head = []
            temp_predicted_dependency = []
            for feature in sent_feature:
                word_id = feature['id']
                head_id = feature['head_id']
                deprel = feature['deprel']
                temp_dependency.append([deprel, head_id, word_id])
                temp_predicted_head.append(head_id)
                temp_predicted_dependency.append(deprel)
            dependency_features.append(temp_dependency)
            head_features.append(temp_predicted_head)
            predicted_dependency_features.append(temp_predicted_dependency)
        dependencies.append(dependency_features)
        predicted_heads.append(head_features)
        predicted_dependencies.append(predicted_dependency_features)

    print("num of examples {}".format(len(dependencies)))

    return dependencies, predicted_heads, predicted_dependencies


def build_dep_tag_vocab(dependencies, vocab_size=1000, min_freq=0):
    counter = Counter()
    for sent_dependency in dependencies:
        for dependency in sent_dependency:
            counter.update(dependency)

    dep_tags = ['<pad>']
    min_freq = max(min_freq, 1)

    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(dep_tags) == vocab_size:
            break
        if word == '<pad>':
            continue
        dep_tags.append(word)

    tag2id = {tag: i for i, tag in enumerate(dep_tags)}
    keys = list(tag2id.keys())
    tags = []
    for i in range(len(tag2id)):
        if i < 2:
            tags.append(keys[i])
            continue
        key = keys[i]
        if key == "root":
            continue
        tags.append(key)

    tag2id = {tag: i for i, tag in enumerate(tags)}

    return dep_tags, tag2id


if __name__ == '__main__':
    language = "english"
    total_dependencies = []
    for name in ["train", "dev", "test"]:
        data_path = "conll_data/{}.{}.depparse.allennlp.jsonlines".format(name, language)
        dependencies, predicted_heads, predicted_dependencies = read_depparse_features(data_path)
        total_dependencies.extend(predicted_dependencies)

    dep_tags, tag2id = build_dep_tag_vocab(total_dependencies)
    print("dependency tag size {}".format(len(tag2id)))
    with open("conll_data/dep_allennlp_vocab.txt", "w") as f:
        for tag, id in tag2id.items():
            f.write(tag + " " + str(id) + "\n")