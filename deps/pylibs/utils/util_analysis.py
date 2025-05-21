import pandas as pd

from pylibs.utils.util_file import FileUtils
from pylibs.uts_metrics.affine.common import ScoreType


def gather_all_files_csv(home, duplicates=False, ext=".csv"):
    """
    读取一个文件夹下面的所有文件.csv输出一个大的csv

    demo:
    from pylibs.utils.util_analysis import gather_all_files_csv
    gather_all_files_csv("./outputs").to_csv("all.csv")

    Parameters
    ----------
    duplicates :

    Returns
    -------

    """
    fs = FileUtils().get_all_files(home, ext)
    data = []
    for f in fs:
        data.append(pd.read_csv(f))
    if duplicates:
        return pd.concat(data, axis=0)
    else:
        return pd.concat(data, axis=0).drop_duplicates()


def get_pandas_connection():
    pass


def get_exp_data() -> pd.DataFrame:
    data = pd.read_sql("""
    select concat_02.SHORT_EXP_NAME,
           concat_02.TRIALCMD_DATASET,
           concat_02.TRAINING_SAMPLE_RATE_FRAC,
           concat_02.AVG_LOSS,
           concat_02.AVG_AUC,
           concat_01.sum_train_time as "TOTAL_TRAINING_TIME(S)",
           concat_02.EXPID,
           concat_02.TRIALJOBID
    from (select SHORT_EXP_NAME,
                 TRIALCMD_DATASET,
                 concat("1/", 1 / TRIALCMD_DATA_SAMPLE_RATE) as TRAINING_SAMPLE_RATE_FRAC,
                 sum(trining_time)                           as sum_train_time,
                 TRIALCMD_DATA_SAMPLE_RATE
          from (select SHORT_EXP_NAME,
                       TRIALCMD_DATASET,
                       TRIALCMD_DATA_SAMPLE_RATE,
                       round((nni_results.metric_train_end - nni_results.metric_train_start)
                                 / 1000000000/60, 4) as trining_time
                from nni_results) b
          group by SHORT_EXP_NAME, TRIALCMD_DATASET, TRIALCMD_DATA_SAMPLE_RATE
          order by SHORT_EXP_NAME, TRIALCMD_DATASET,
                   TRIALCMD_DATA_SAMPLE_RATE desc) concat_01
             left join (select r1.TRIALCMD_DATASET,
                               concat("1/", 1 / r1.TRIALCMD_DATA_SAMPLE_RATE) as TRAINING_SAMPLE_RATE_FRAC,
                               r1.AVG_LOSS,
                               r1.AVG_AUC,
                               r1.EXPID,
                               r1.TRIALJOBID,
                               r1.SHORT_EXP_NAME,
                               r1.TRIALCMD_DATA_SAMPLE_RATE
                        from (select SHORT_EXP_NAME,
                                     TRIALCMD_DATASET,
                                     EXPID,
                                     TRIALJOBID,
                                     TRIALCMD_DATA_SAMPLE_RATE,
                                     round((METRIC_TEST_AUC_0 + METRIC_TEST_AUC_1 + METRIC_TEST_AUC_2 + METRIC_TEST_AUC_3 +
                                            METRIC_TEST_AUC_4 +
                                            METRIC_TEST_AUC_5 + METRIC_TEST_AUC_6 + METRIC_TEST_AUC_7 + METRIC_TEST_AUC_8 +
                                            METRIC_TEST_AUC_9) /
                                           10,
                                           4)
                                         as AVG_AUC,
                                     round((METRIC_TEST_LOSS_0 + METRIC_TEST_LOSS_1 + METRIC_TEST_LOSS_2 +
                                            METRIC_TEST_LOSS_3 +
                                            nni_results.METRIC_TEST_LOSS_4 + METRIC_TEST_LOSS_5 + METRIC_TEST_LOSS_6 +
                                            METRIC_TEST_LOSS_7 +
                                            METRIC_TEST_LOSS_8 + METRIC_TEST_LOSS_9) / 10, 4)
                                         as AVG_LOSS
                              from nni_results) as r1
                        where r1.AVG_LOSS is not null
                          and not exists(select 1
                                         from (select SHORT_EXP_NAME,
                                                      TRIALCMD_DATASET,
                                                      EXPID,
                                                      TRIALJOBID,
                                                      TRIALCMD_DATA_SAMPLE_RATE,
                                                      round((METRIC_TEST_AUC_0 + METRIC_TEST_AUC_1 + METRIC_TEST_AUC_2 +
                                                             METRIC_TEST_AUC_3 +
                                                             METRIC_TEST_AUC_4 +
                                                             METRIC_TEST_AUC_5 + METRIC_TEST_AUC_6 + METRIC_TEST_AUC_7 +
                                                             METRIC_TEST_AUC_8 +
                                                             METRIC_TEST_AUC_9) /
                                                            10,
                                                            4)
                                                          as AVG_AUC,
                                                      round((METRIC_TEST_LOSS_0 + METRIC_TEST_LOSS_1 + METRIC_TEST_LOSS_2 +
                                                             METRIC_TEST_LOSS_3 +
                                                             nni_results.METRIC_TEST_LOSS_4 + METRIC_TEST_LOSS_5 +
                                                             METRIC_TEST_LOSS_6 +
                                                             METRIC_TEST_LOSS_7 +
                                                             METRIC_TEST_LOSS_8 + METRIC_TEST_LOSS_9) / 10, 4)
                                                          as AVG_LOSS
                                               from nni_results) r2
                                         where r1.SHORT_EXP_NAME = r2.SHORT_EXP_NAME
                                           and r1.TRIALCMD_DATASET = r2.TRIALCMD_DATASET
                                           and r1.TRIALCMD_DATA_SAMPLE_RATE = r2.TRIALCMD_DATA_SAMPLE_RATE
                                           and r1.AVG_LOSS > r2.AVG_LOSS
                            )
                        order by r1.SHORT_EXP_NAME, r1.TRIALCMD_DATASET, r1.TRIALCMD_DATA_SAMPLE_RATE DESC) concat_02
                       on concat_01.SHORT_EXP_NAME = concat_02.SHORT_EXP_NAME and
                          concat_01.TRIALCMD_DATASET = concat_02.TRIALCMD_DATASET and
                          concat_01.TRIALCMD_DATA_SAMPLE_RATE = concat_02.TRIALCMD_DATA_SAMPLE_RATE
    order by concat_01.SHORT_EXP_NAME, concat_01.TRIALCMD_DATASET, concat_01.TRIALCMD_DATA_SAMPLE_RATE DESC; 
    """, get_pandas_connection())
    return data


def get_exp_data_v2():
    d1 = get_exp_data_v1(score_type=ScoreType.Pointwise)
    d2 = get_exp_data_v1(score_type=ScoreType.PointAdjusted)
    keys = ['TRIALCMD_DATA_ID', 'TRAINING_SAMPLE_RATE_FRAC']
    target = d1.merge(d2, left_on=keys, right_on=keys)
    d3 = get_exp_data_v1(score_type=ScoreType.RevisedPointAdjusted)
    target = target.merge(d3, left_on=keys, right_on=keys)
    d4 = get_exp_data_v1(score_type=ScoreType.Affiliation)
    target = target.merge(d4, left_on=keys, right_on=keys)
    return target


def get_exp_data_v1(score_type) -> pd.DataFrame:
    if score_type == ScoreType.Pointwise:
        metric = "METRIC_TEST_POINT_WISE_F1"
    elif score_type == ScoreType.PointAdjusted:
        metric = "METRIC_TEST_POINT_ADJUSTED_F1"
    elif score_type == ScoreType.RevisedPointAdjusted:
        metric = "METRIC_TEST_REVISED_POINT_ADJUSTED_F1"
    elif score_type == ScoreType.Affiliation:
        metric = "METRIC_TEST_AFFILIATION_F1"
    else:
        raise TypeError(f"Unsupported score type {score_type}")

    data = pd.read_sql(f"""select r1.TRIALCMD_DATASET,
       r1.TRIALCMD_DATA_ID,
       concat("1/", 1 / r1.TRIALCMD_DATA_SAMPLE_RATE) as TRAINING_SAMPLE_RATE_FRAC,
       {metric},
       r1.EXPID,
       r1.TRIALJOBID,
       r1.TRIALCMD_DATA_SAMPLE_RATE
from (select * from nni_results) as r1
where r1.EXPINFO_EXPERIMENTNAME like  'EXP_01_1672131902%%'
  and {metric} is not null
  and not exists(select 1
                 from (select TRIALCMD_DATASET
                            , TRIALCMD_DATA_ID
                            , {metric}
                            , TRIALCMD_DATA_SAMPLE_RATE
                       from nni_results) r2
                 where r1.TRIALCMD_DATASET = r2.TRIALCMD_DATASET
                   and r1.TRIALCMD_DATA_ID = r2.TRIALCMD_DATA_ID
                   and r1.TRIALCMD_DATA_SAMPLE_RATE = r2.TRIALCMD_DATA_SAMPLE_RATE
                   and r1.{metric} < r2.{metric}
    )
order by r1.TRIALCMD_DATASET, r1.TRIALCMD_DATA_ID, r1.TRIALCMD_DATA_SAMPLE_RATE desc """, get_pandas_connection())
    # exp_index = data.loc[:, ['TRIALCMD_DATA_ID', 'TRAINING_SAMPLE_RATE_FRAC']]
    # data.exp_index = pd.MultiIndex.from_frame(exp_index)
    return data
