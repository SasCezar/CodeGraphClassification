import json

from hydra import initialize, compose


def create_example():
    with initialize(version_base=None, config_path="../../src/conf/"):
        cfg = compose(config_name='annotation.yaml', overrides=["local=default"])
    project = 'Waikato|weka-3.8.json'

    file = 'weka/src/main/java/weka/classifiers/functions/SimpleLogistic.java'

    table = []
    keywords_file = f'{cfg.base_path}/data/processed/content/identifiers/{project}'
    with open(keywords_file) as inf:
        obj = json.load(inf)

    file_terms = obj['content'][file]
    print(file_terms)


if __name__ == '__main__':
    create_example()
