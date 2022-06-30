import pandas as pd


def dataset2csv():
    dataset_path = '/home/sasce/PycharmProjects/GitHubClassificationDataset/data/classification_dataset_lang.json'
    dataset = pd.read_json(dataset_path, lines=True)
    dataset = dataset[['full_name', 'language', 'labels', 'levels', 'topics']]
    dataset.to_csv('/home/sasce/PycharmProjects/CodeGraphClassification/data/raw/classification_dataset_lang.csv',
                   index=False)


if __name__ == '__main__':
    dataset2csv()
