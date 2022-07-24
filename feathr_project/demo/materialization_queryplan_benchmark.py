import time
import argparse

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, date_sub
from pyspark.storagelevel import StorageLevel


def conf_setup(app_string):
    conf = SparkConf()
    conf.setAppName(app_string)
    conf.set("spark.local.dir", "/home/ruiliu/Develop/feathr-pqo/feathr_project/demo/persist")
    spark_session = SparkSession.builder.config(conf=conf).getOrCreate()

    spark_context = spark_session.sparkContext

    return spark_session, spark_context


def get_dataset(dataset_scale_factor):
    if dataset_scale_factor == "small":
        user_profile_path = "dataset/mock_user_profile.csv"
        purchase_history_path = "dataset/mock_purchase_history.csv"
        observation_path = "dataset/mock_observation.csv"
    elif dataset_scale_factor == "medium":
        user_profile_path = "dataset/user_profile_20.csv"
        purchase_history_path = "dataset/purchase_history_100K.csv"
        observation_path = "dataset/observation_100K.csv"
    else:
        user_profile_path = "dataset/user_profile_100.csv"
        purchase_history_path = "dataset/purchase_history_50M.csv"
        observation_path = "dataset/observation_50M.csv"

    return user_profile_path, purchase_history_path, observation_path


def query_materialization(dataset_scale):
    user_profile_path, purchase_history_path, observation_path = get_dataset(dataset_scale)

    spark_sess, spark_ctx = conf_setup("query_precompute")

    user_profile = spark_sess.read.csv(user_profile_path, header=True)
    purchase_history = spark_sess.read.csv(purchase_history_path, header=True)
    observation = spark_sess.read.csv(observation_path, header=True)

    obs_purchase_pit = (observation.join(purchase_history, on=["user_id"], how="inner")
                        .filter((purchase_history.purchase_date >= date_sub(observation.event_timestamp, 200))
                                & (purchase_history.purchase_date < observation.event_timestamp)))

    obs_purchase_pit_cache = obs_purchase_pit.cache()

    obs_purchase_profile = obs_purchase_pit.join(user_profile, on=["user_id"], how="inner")

    total_purchase = (obs_purchase_profile.dropDuplicates(["purchase_date"])
                      .groupby("user_id")
                      .agg(sum("purchase_amount").alias("total_purchase_pit")))

    total_purchase_cache = total_purchase.cache()

    # query_result_pit = obs_purchase_profile.join(total_purchase, on=["user_id"], how="inner")
    # print(query_result_pit.count())

    ################################

    spark_sess, spark_ctx = conf_setup("query_materialization")

    user_profile = spark_sess.read.csv(user_profile_path, header=True)
    purchase_history = spark_sess.read.csv(purchase_history_path, header=True)
    observation = spark_sess.read.csv(observation_path, header=True)

    start = time.perf_counter()
    obs_purchase_pit = (observation.join(purchase_history, on=["user_id"], how="inner")
                        .filter((purchase_history.purchase_date >= date_sub(observation.event_timestamp, 300))
                                & (purchase_history.purchase_date < date_sub(observation.event_timestamp, 200))))

    obs_purchase_profile = obs_purchase_pit.join(user_profile, on=["user_id"], how="inner")

    total_purchase = (obs_purchase_profile.dropDuplicates(["purchase_date"])
                      .groupby("user_id")
                      .agg(sum("purchase_amount").alias("total_purchase_pit")))

    total_purchase_all = (total_purchase.union(total_purchase_cache)
                                        .groupby("user_id")
                                        .agg(sum("total_purchase_pit").alias("total_purchase_pit")))

    obs_purchase_profile_all = obs_purchase_pit.union(obs_purchase_pit_cache)

    query_result_materialization = obs_purchase_profile_all.join(total_purchase_all, on=["user_id"], how="inner")
    print(query_result_materialization.count())
    end = time.perf_counter()
    proc_time = end - start
    print("Query process time with materialization is {}".format(proc_time))


def query_no_materialization(dataset_scale):
    user_profile_path, purchase_history_path, observation_path = get_dataset(dataset_scale)

    spark_sess, spark_ctx = conf_setup("query_no_materialization")

    user_profile = spark_sess.read.csv(user_profile_path, header=True)
    purchase_history = spark_sess.read.csv(purchase_history_path, header=True)
    observation = spark_sess.read.csv(observation_path, header=True)

    start = time.perf_counter()
    obs_purchase_pit = (observation.join(purchase_history, on=["user_id"], how="inner")
                        .filter((purchase_history.purchase_date >= date_sub(observation.event_timestamp, 300))
                                & (purchase_history.purchase_date < observation.event_timestamp)))

    total_purchase = (obs_purchase_pit.dropDuplicates(["purchase_date"])
                      .groupby("user_id")
                      .agg(sum("purchase_amount").alias("total_purchase_pit")))

    total_purchase.show()

    obs_purchase_profile = obs_purchase_pit.join(user_profile, on=["user_id"], how="inner")

    obs_purchase_profile.show(1000)

    # query_results.show(100)
    '''
    total_purchase = (obs_purchase_profile.dropDuplicates(["purchase_date"])
                      .groupby("user_id")
                      .agg(sum("purchase_amount").alias("total_purchase_pit")))

    query_result_no_materialization = obs_purchase_profile.join(total_purchase, on=["user_id"], how="inner")
    print(query_result_no_materialization.count())
    '''
    end = time.perf_counter()
    proc_time = end - start
    print("Query process time without materialization is {}".format(proc_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_scale",
                        action="store",
                        type=str,
                        default="small",
                        choices=["small", "medium", "large"],
                        help="indicate dataset scale")

    args = parser.parse_args()
    dataset_scale = args.dataset_scale

    # query_materialization(dataset_scale)

    query_no_materialization(dataset_scale)


if __name__ == "__main__":
    main()
