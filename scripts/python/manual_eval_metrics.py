from collections import defaultdict, Counter
from os.path import join

import hydra
import numpy as np
import pandas as pd
from agreement import krippendorffs_alpha, cohens_kappa
from agreement.utils.transform import pivot_table_frequency
from omegaconf import DictConfig


def read_annotations(path):
    # Load Excel file into a Pandas ExcelFile object
    xlsx_file = pd.ExcelFile(path)

    # Read all sheets except the first one into a dictionary of dataframes with column headers set to the first row
    dfs_dict = pd.read_excel(xlsx_file, sheet_name=xlsx_file.sheet_names[1:], header=0)

    # Rename the first column header to "annot_id" for all dataframes
    for df in dfs_dict.values():
        df.rename(columns={df.columns[0]: 'annot_id'}, inplace=True)
        df.rename(columns={df.columns[-1]: 'annotation'}, inplace=True)

    # Concatenate all dataframes in the dictionary into a single dataframe
    result_df = pd.concat(dfs_dict.values(), ignore_index=True)

    return result_df


def project_group_annotations(df, _):
    df_grouped = df.groupby('annot_id').agg({'project': 'first',
                                             'url': 'first',
                                             'label': 'first',
                                             'annotator': list,
                                             'annotation': [('annotation1', lambda x: x.iloc[0]),
                                                            ('annotation2', lambda x: x.iloc[1])]}).reset_index()

    df_grouped.columns = list(map(lambda x: x[0] if x[1] in ['list', 'first'] else x[1], df_grouped.columns.values))
    df_grouped.columns = ['annot_id'] + list(df_grouped.columns.values)[1:]
    return df_grouped


def others_group_annotations(df, level='file'):
    df_grouped = df.groupby('annot_id').agg({
        'project': 'first',
        'url': 'first',
        level: 'first',
        'label_1': 'first',
        'label_2': 'first',
        'label_3': 'first',
        'annotator': list,
        'annotation': [('annotation1', lambda x: x.iloc[0]),
                       ('annotation2', lambda x: x.iloc[1])]}).reset_index()

    df_grouped.columns = list(map(lambda x: x[0] if x[1] in ['list', 'first'] else x[1], df_grouped.columns.values))
    df_grouped.columns = ['annot_id'] + list(df_grouped.columns.values)[1:]
    return df_grouped


def create_annot_table(df):
    question_id = df['annot_id']
    annotator_id = df['annotator']
    annotation = df['annotation']

    annot_table = np.column_stack((question_id, annotator_id, annotation))
    return annot_table


def load_disagreement(path):
    df = pd.read_csv(path, header=0)
    df.rename(columns={df.columns[-1]: 'resolved'}, inplace=True)
    df = df[df.columns.intersection(['annot_id', 'resolved'])]
    return df


def join_disagreement(df, disagreement_df):
    df = df.merge(disagreement_df, how='left', left_on='annot_id', right_on='annot_id')

    df['final'] = df['resolved']
    df['final'].fillna(df['annotation1'], inplace=True)
    df['final'] = df['final'].astype('uint8')
    return df


def get_agreement(df):
    answer_table = pivot_table_frequency(np.array(df['annot_id']),
                                         np.array(df['annotation'].replace('-', '0').fillna(0)).astype('int64'))
    user_table = pivot_table_frequency(np.array(df['annotator']),
                                       np.array(df['annotation'].replace('-', '0').fillna(0)).astype('int64'))
    alpha = krippendorffs_alpha(answer_table)
    kappa = cohens_kappa(answer_table, user_table)

    return alpha, kappa


@hydra.main(config_path="../../src/conf", config_name="annotation", version_base="1.2")
def evaluation(cfg: DictConfig):
    files = {
        'project': join(cfg.out_path, 'processed/manual_eval/project_human_eval_best-voting.xlsx'),
        'package': join(cfg.out_path, 'processed/manual_eval/package_human_eval_best-voting.xlsx'),
        'file': join(cfg.out_path, 'processed/manual_eval/file_human_eval_best-voting.xlsx')
    }

    grouping = {'project': project_group_annotations, 'file': others_group_annotations,
                'package': others_group_annotations}

    disagreement_files = {'project': join(cfg.out_path, 'processed/manual_eval/project_human_disagreements.csv'),
                          'package': join(cfg.out_path, 'processed/manual_eval/package_human_disagreements.csv'),
                          'file': join(cfg.out_path, 'processed/manual_eval/file_human_disagreements.csv')
                          }

    scores = defaultdict(lambda: Counter())

    agreements = []
    results = []

    for level in files:
        file = files[level]
        df = read_annotations(file)
        df["annotation"] = df["annotation"].replace('-', 0).fillna(0).astype('int64')

        alpha, kappa = get_agreement(df)
        agreements.extend([(level, 'alpha', alpha), (level, 'kappa', kappa)])

        disagreement_df = load_disagreement(disagreement_files[level])
        grouped_df = grouping[level](df, level)
        resolved_df = join_disagreement(grouped_df, disagreement_df)
        resolved_df.to_csv(join(cfg.out_path, f'processed/manual_eval/{level}_annotation_combined.csv'), index=False)
        print(level, kappa, len(disagreement_df) / len(grouped_df), len(grouped_df))

        scores[level].update(resolved_df['final'])

        # annot_table = create_annot_table(df)

        total = sum([x[1] for x in scores[level].most_common()])

        for x, num in scores[level].most_common():
            group = 'Correct' if x else 'Incorrect'
            results.append((level, group, x, num, num / total))

        if level == 'project':
            result = resolved_df.groupby('project', as_index=False).agg({'final': 'sum'})['final']
            print(np.mean(result))
            count = Counter()
            count.update(result)
            result = pd.DataFrame(count.most_common(), columns=['num', 'count'])
            result.to_csv(f'{cfg.base_path}/output/stats/LF_project_new_topics.csv',
                          index=False)

    results_df = pd.DataFrame(results, columns=['level', 'group', 'label', 'num', 'percent'])
    results_df.to_csv(f'{cfg.base_path}/output/stats/human_eval_results.csv',
                      index=False)

    agreements_df = pd.DataFrame(agreements, columns=['level', 'metric', 'score'])
    agreements_df.to_csv(f'{cfg.base_path}/output/stats/human_eval_agreement.csv',
                         index=False)


if __name__ == '__main__':
    evaluation()
