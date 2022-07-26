import time
import argparse

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import min, date_sub, lit, to_date
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


def query_optimized(user_profile, purchase_history, observation):
    min_event_timestamp = observation.agg(min("event_timestamp")).collect()[0][0]

    purchase_history_filter = purchase_history.filter(
        purchase_history.purchase_date >= date_sub(to_date(lit(min_event_timestamp)), 300))

    obs_purchase_pit = observation.join(purchase_history_filter, on=["user_id"], how="inner").filter(
        (purchase_history.purchase_date >= date_sub(observation.event_timestamp, 300))
        & (purchase_history.purchase_date < date_sub(observation.event_timestamp, 100)))

    obs_purchase_profile = obs_purchase_pit.join(user_profile, on=["user_id"], how="inner")

    obs_purchase_profile_ckpt = obs_purchase_profile.checkpoint(eager=True)

    query_start = time.perf_counter()

    obs_purchase_join = observation.join(purchase_history_filter, on=["user_id"], how="inner")

    obs_purchase_filter = obs_purchase_join.filter(
        (purchase_history.purchase_date >= date_sub(observation.event_timestamp, 100))
        & (purchase_history.purchase_date < observation.event_timestamp))

    obs_purchase_profile = obs_purchase_filter.join(user_profile, on=["user_id"], how="inner")

    query_optimized_result = obs_purchase_profile.union(obs_purchase_profile_ckpt)
    print("Tuples after union ckpt with results: {}".format(query_optimized_result.count()))

    query_end = time.perf_counter()
    print("## Overall time is {}".format(query_end - query_start))

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

    return query_optimized_result


def query_original(user_profile, purchase_history, observation):
    query_start = time.perf_counter()

    obs_purchase_join = observation.join(purchase_history, on=["user_id"], how="inner")

    obs_purchase_filter = obs_purchase_join.filter(
        (purchase_history.purchase_date >= date_sub(observation.event_timestamp, 300))
        & (purchase_history.purchase_date < observation.event_timestamp))

    query_original_result = obs_purchase_filter.join(user_profile, on=["user_id"], how="inner")
    print("Tuples after joining filtering results and user profiles: {}".format(query_original_result.count()))

    query_end = time.perf_counter()
    print("## Overall time is {:.2f}s".format(query_end - query_start))

    return query_original_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_scale",
                        action="store",
                        type=str,
                        default="small",
                        choices=["small", "medium", "large"],
                        help="indicate dataset scale")
    parser.add_argument('--opt', action='store_true')

    args = parser.parse_args()
    dataset_scale = args.dataset_scale
    # query_opt = args.opt

    spark_sess, spark_ctx = conf_setup("queryplan_optimization")

    user_profile_path, purchase_history_path, observation_path = get_dataset(dataset_scale)

    user_profile_dataset = spark_sess.read.csv(user_profile_path, header=True)
    purchase_history_dataset = spark_sess.read.csv(purchase_history_path, header=True)
    observation_dataset = spark_sess.read.csv(observation_path, header=True)

    print("## Processing Query without Optimization ##")
    original_results = query_original(user_profile_dataset, purchase_history_dataset, observation_dataset)

    print("## Processing Query with Optimization ##")
    optimized_results = query_optimized(user_profile_dataset, purchase_history_dataset, observation_dataset)

    if original_results.subtract(optimized_results).count() == 0:
        print("[CHECK]: The results of with and without optimization are SAME")
    else:
        print("[CHECK]: The results of with and without optimization are DIFFERENT")


if __name__ == "__main__":
    main()
