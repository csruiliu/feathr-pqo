import time
import argparse

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import datediff, sum
from pyspark.storagelevel import StorageLevel


def conf_setup():
    conf = SparkConf()
    conf.setAppName("materialization_queryplan")
    conf.set("spark.local.dir", "/home/ruiliu/Develop/feathr-pqo/feathr_project/demo/persist")
    spark_session = SparkSession.builder.config(conf=conf).getOrCreate()

    spark_context = spark_session.sparkContext

    return spark_session, spark_context


def get_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_scale",
                        action="store",
                        type=str,
                        default="small",
                        choices=["small", "medium", "large"],
                        help="indicate dataset scale")

    args = parser.parse_args()
    dataset_scale = args.dataset_scale
    if dataset_scale == "small":
        user_profile_path = "dataset/mock_user_profile.csv"
        purchase_history_path = "dataset/mock_purchase_history.csv"
        observation_path = "dataset/mock_observation.csv"
    elif dataset_scale == "medium":
        print("ssss")
        user_profile_path = "dataset/user_profile_20.csv"
        purchase_history_path = "dataset/purchase_history_1M.csv"
        observation_path = "dataset/observation_1M.csv"
    else:
        user_profile_path = "dataset/user_profile_100.csv"
        purchase_history_path = "dataset/purchase_history_50M.csv"
        observation_path = "dataset/observation_50M.csv"

    return user_profile_path, purchase_history_path, observation_path


def main():
    user_profile_path, purchase_history_path, observation_path = get_dataset()

    spark_sess, spark_ctx = conf_setup()

    user_profile = spark_sess.read.csv(user_profile_path, header=True)
    purchase_history = spark_sess.read.csv(purchase_history_path, header=True)
    observation = spark_sess.read.csv(observation_path, header=True)

    start = time.perf_counter()
    obs_purchase_pit = (observation.join(purchase_history,
                                         on=[observation.user_id == purchase_history.user_id],
                                         how="inner")
                                    .drop(purchase_history.user_id)
                                    .filter((datediff(observation.event_timestamp, purchase_history.purchase_date) <= 60)
                                            & (observation.event_timestamp > purchase_history.purchase_date)))

    obs_purchase_profile = obs_purchase_pit.join(user_profile,
                                                 on=[obs_purchase_pit.user_id == user_profile.user_id],
                                                 how="inner").drop(user_profile.user_id)

    total_purchase = (obs_purchase_profile.dropDuplicates(["purchase_date"])
                                          .groupby("user_id")
                                          .agg(sum("purchase_amount")
                                          .alias("user_total_purchase_60days")))

    query_result = obs_purchase_profile.join(total_purchase,
                                             on=[obs_purchase_profile.user_id == total_purchase.user_id],
                                             how="inner")
    query_result.show()
    end = time.perf_counter()
    proc_time = end - start
    print("All the process time without materialization: {}".format(proc_time))

    start = time.perf_counter()
    obs_purchase_pit = (observation.join(purchase_history,
                                         on=[observation.user_id == purchase_history.user_id],
                                         how="inner")
                        .drop(purchase_history.user_id)
                        .filter((datediff(observation.event_timestamp, purchase_history.purchase_date) <= 90)
                                & (observation.event_timestamp > purchase_history.purchase_date)))

    obs_purchase_profile = obs_purchase_pit.join(user_profile,
                                                 on=[obs_purchase_pit.user_id == user_profile.user_id],
                                                 how="inner").drop(user_profile.user_id)

    total_purchase = (obs_purchase_profile.dropDuplicates(["purchase_date"])
                      .groupby("user_id")
                      .agg(sum("purchase_amount")
                           .alias("user_total_purchase_60days")))

    query_result = obs_purchase_profile.join(total_purchase,
                                             on=[obs_purchase_profile.user_id == total_purchase.user_id],
                                             how="inner")
    query_result.show()
    end = time.perf_counter()
    proc_time = end - start
    print("All the process time without materialization: {}".format(proc_time))



if __name__ == "__main__":
    main()
