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
    spark_context.setCheckpointDir("/home/ruiliu/Develop/feathr-pqo/feathr_project/demo/ckpt")

    return spark_session, spark_context


def get_dataset(dataset_scale_factor):
    if dataset_scale_factor == "small":
        user_profile_path = "dataset/mock_user_profile.csv"
        purchase_history_path = "dataset/mock_purchase_history.csv"
        observation_path = "dataset/mock_observation.csv"
    elif dataset_scale_factor == "medium":
        user_profile_path = "dataset/user_profile_50.csv"
        purchase_history_path = "dataset/purchase_history_100K.csv"
        observation_path = "dataset/observation_100K.csv"
    else:
        user_profile_path = "dataset/user_profile_250.csv"
        purchase_history_path = "dataset/purchase_history_500K.csv"
        observation_path = "dataset/observation_500K.csv"

    return user_profile_path, purchase_history_path, observation_path


def query_materialization(dataset_scale):
    user_profile_path, purchase_history_path, observation_path = get_dataset(dataset_scale)

    spark_sess, spark_ctx = conf_setup("query_materialization")

    user_profile = spark_sess.read.csv(user_profile_path, header=True)
    purchase_history = spark_sess.read.csv(purchase_history_path, header=True)
    observation = spark_sess.read.csv(observation_path, header=True)

    obs_purchase_pit = observation.join(purchase_history, on=["user_id"], how="inner").filter(
        (purchase_history.purchase_date >= date_sub(observation.event_timestamp, 600))
        & (purchase_history.purchase_date < date_sub(observation.event_timestamp, 100)))
    '''
    obs_purchase_pit = observation.join(purchase_history, on=["user_id"], how="inner").filter(
        (purchase_history.purchase_date >= date_sub(observation.event_timestamp, 400))
        & (purchase_history.purchase_date < observation.event_timestamp))
    '''
    obs_purchase_profile = obs_purchase_pit.join(user_profile, on=["user_id"], how="inner")

    obs_purchase_profile_ckpt = obs_purchase_profile.checkpoint(eager=True)

    ################################

    # spark_sess, spark_ctx = conf_setup("query_materialization")

    # user_profile = spark_sess.read.csv(user_profile_path, header=True)
    # purchase_history = spark_sess.read.csv(purchase_history_path, header=True)
    # observation = spark_sess.read.csv(observation_path, header=True)

    global_start = time.perf_counter()

    start = time.perf_counter()
    obs_purchase_join = observation.join(purchase_history, on=["user_id"], how="inner")
    print("Tuples after joining observation and purchase history: {}".format(obs_purchase_join.count()))
    end = time.perf_counter()
    print("## Processing time of joining observation and purchase history: {:.2f}s".format(end - start))

    start = time.perf_counter()
    obs_purchase_filter = obs_purchase_join.filter(
        (purchase_history.purchase_date >= date_sub(observation.event_timestamp, 100))
        & (purchase_history.purchase_date < observation.event_timestamp))
    '''
    obs_purchase_filter = obs_purchase_join.filter(
        (purchase_history.purchase_date >= date_sub(observation.event_timestamp, 600))
        & (purchase_history.purchase_date < date_sub(observation.event_timestamp, 400)))
    '''
    print("Tuples after getting data in the time window: {}".format(obs_purchase_filter.count()))
    end = time.perf_counter()
    print("## Processing time of getting data in the time window: {:.2f}s".format(end - start))

    start = time.perf_counter()
    obs_purchase_profile = obs_purchase_filter.join(user_profile, on=["user_id"], how="inner")
    print("Tuples after joining filtering results and user profiles: {}".format(obs_purchase_profile.count()))
    end = time.perf_counter()
    print("## Processing time of joining filtering results and user profiles is {:.2f}s".format(end - start))

    start = time.perf_counter()
    query_result_materialization = obs_purchase_profile.union(obs_purchase_profile_ckpt)
    print("Tuples after union ckpt with results: {}".format(query_result_materialization.count()))
    end = time.perf_counter()
    print("## Processing time of union ckpt with results {:.2f}s".format(end - start))

    global_end = time.perf_counter()
    print("## Overall time is {}".format(global_end - global_start))

    '''
    total_purchase = (obs_purchase_profile.dropDuplicates(["purchase_date"])
                      .groupby("user_id")
                      .agg(sum("purchase_amount").alias("total_purchase_pit")))
    
    total_purchase_all = (total_purchase.union(total_purchase_cache)
                                        .groupby("user_id")
                                        .agg(sum("total_purchase_pit").alias("total_purchase_pit")))

    obs_purchase_profile_all = obs_purchase_pit.union(obs_purchase_pit_cache)

    query_result_materialization = obs_purchase_profile_all.join(total_purchase_all, on=["user_id"], how="inner")
    '''


def query_no_materialization(dataset_scale):
    user_profile_path, purchase_history_path, observation_path = get_dataset(dataset_scale)

    spark_sess, spark_ctx = conf_setup("query_no_materialization")

    user_profile = spark_sess.read.csv(user_profile_path, header=True)
    purchase_history = spark_sess.read.csv(purchase_history_path, header=True)
    observation = spark_sess.read.csv(observation_path, header=True)

    global_start = time.perf_counter()

    start = time.perf_counter()
    obs_purchase_join = observation.join(purchase_history, on=["user_id"], how="inner")
    print("Tuples after joining observation and purchase history: {}".format(obs_purchase_join.count()))
    end = time.perf_counter()
    print("## Processing time of joining observation and purchase history: {:.2f}s".format(end - start))

    start = time.perf_counter()
    obs_purchase_filter = obs_purchase_join.filter(
        (purchase_history.purchase_date >= date_sub(observation.event_timestamp, 600))
        & (purchase_history.purchase_date < observation.event_timestamp))
    print("Tuples after getting data in the time window: {}".format(obs_purchase_filter.count()))
    end = time.perf_counter()
    print("## Processing time of getting data in the time window: {:.2f}s".format(end - start))

    start = time.perf_counter()
    query_result_no_materialization = obs_purchase_filter.join(user_profile, on=["user_id"], how="inner")
    print("Tuples after joining filtering results and user profiles: {}".format(query_result_no_materialization.count()))
    end = time.perf_counter()
    print("## Processing time of joining filtering results and user profiles is {:.2f}s".format(end - start))

    global_end = time.perf_counter()
    print("## Overall time is {:.2f}s".format(global_end - global_start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_scale",
                        action="store",
                        type=str,
                        default="small",
                        choices=["small", "medium", "large"],
                        help="indicate dataset scale")
    parser.add_argument('--mv', action='store_true')

    args = parser.parse_args()
    dataset_scale = args.dataset_scale
    materialization = args.mv
    if materialization:
        print("## Processing Query with Materialization ##")
        query_materialization(dataset_scale)
    else:
        print("## Processing Query without Materialization ##")
        query_no_materialization(dataset_scale)


if __name__ == "__main__":
    main()
