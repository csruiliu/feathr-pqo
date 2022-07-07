import os
import glob
import time
import tempfile
import pandas as pd
import pandavro as pdx
from datetime import datetime, timedelta

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from feathr import FeathrClient, FeatureQuery, ObservationSettings
from feathr import Feature, DerivedFeature, FeatureAnchor
from feathr import BackfillTime, MaterializationSettings
from feathr import BOOLEAN, FLOAT, INT32, ValueType
from feathr import INPUT_CONTEXT, HdfsSource
from feathr import WindowAggTransformation
from feathr import TypedKey
from feathr import RedisSink, HdfsSink
from feathr.job_utils import get_result_df


def config_runtime():
    yaml_config = """
    # Please refer to https://github.com/linkedin/feathr/blob/main/feathr_project/feathrcli/data/feathr_user_workspace/feathr_config.yaml for explanations on the meaning of each field.
    api_version: 1
    project_config:
      project_name: 'feathr_getting_started'
      required_environment_variables:
        - 'REDIS_PASSWORD'
        - 'AZURE_CLIENT_ID'
        - 'AZURE_TENANT_ID'
        - 'AZURE_CLIENT_SECRET'
    offline_store:
      adls:
        adls_enabled: true
      wasb:
        wasb_enabled: true
      s3:
        s3_enabled: false
        s3_endpoint: 's3.amazonaws.com'
      jdbc:
        jdbc_enabled: false
        jdbc_database: 'feathrtestdb'
        jdbc_table: 'feathrtesttable'
      snowflake:
        url: "dqllago-ol19457.snowflakecomputing.com"
        user: "feathrintegration"
        role: "ACCOUNTADMIN"
    spark_config:
      spark_cluster: 'azure_synapse'
      spark_result_output_parts: '1'
      azure_synapse:
        dev_url: 'https://feathrazuretest3synapse.dev.azuresynapse.net'
        pool_name: 'spark3'
        workspace_dir: 'abfss://feathrazuretest3fs@feathrazuretest3storage.dfs.core.windows.net/feathr_getting_started'
        executor_size: 'Small'
        executor_num: 4
        feathr_runtime_location: wasbs://public@azurefeathrstorage.blob.core.windows.net/feathr-assembly-LATEST.jar
      databricks:
        workspace_instance_url: 'https://adb-2474129336842816.16.azuredatabricks.net'
        config_template: {'run_name':'','new_cluster':{'spark_version':'9.1.x-scala2.12','node_type_id':'Standard_D3_v2','num_workers':2,'spark_conf':{}},'libraries':[{'jar':''}],'spark_jar_task':{'main_class_name':'','parameters':['']}}
        work_dir: 'dbfs:/feathr_getting_started'
        feathr_runtime_location: https://azurefeathrstorage.blob.core.windows.net/public/feathr-assembly-LATEST.jar
    online_store:
      redis:
        host: 'feathrazuretest3redis.redis.cache.windows.net'
        port: 6380
        ssl_enabled: True
    feature_registry:
      purview:
        type_system_initialization: true
        purview_name: 'feathrazuretest3-purview1'
        delimiter: '__'
    """
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    with open(tmp.name, "w") as text_file:
        text_file.write(yaml_config)

    return tmp


def config_feathr():
    #############################
    # Prerequisite Configuration
    #############################
    # resource prefix
    resource_prefix = "feathrpqoplus"

    # Get all the required credentials from Azure Key Vault
    key_vault_name = resource_prefix + "kv"
    synapse_workspace_url = resource_prefix + "syws"
    adls_account = resource_prefix + "dls"
    adls_fs_name = resource_prefix + "fs"
    purview_name = resource_prefix + "purview"
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    client = SecretClient(vault_url=key_vault_uri, credential=credential)
    secretName = "FEATHR-ONLINE-STORE-CONN"
    retrieved_secret = client.get_secret(secretName).value

    # Get redis credentials; This is to parse Redis connection string.
    redis_port = retrieved_secret.split(',')[0].split(":")[1]
    redis_host = retrieved_secret.split(',')[0].split(":")[0]
    redis_password = retrieved_secret.split(',')[1].split("password=", 1)[1]
    redis_ssl = retrieved_secret.split(',')[2].split("ssl=", 1)[1]

    # Set the resource link
    os.environ['spark_config__azure_synapse__dev_url'] = f'https://{synapse_workspace_url}.dev.azuresynapse.net'
    os.environ['spark_config__azure_synapse__pool_name'] = 'spark31'
    os.environ[
        'spark_config__azure_synapse__workspace_dir'] = f'abfss://{adls_fs_name}@{adls_account}.dfs.core.windows.net/feathr_project'
    os.environ['feature_registry__purview__purview_name'] = f'{purview_name}'
    os.environ['online_store__redis__host'] = redis_host
    os.environ['online_store__redis__port'] = redis_port
    os.environ['online_store__redis__ssl_enabled'] = redis_ssl
    os.environ['REDIS_PASSWORD'] = redis_password
    os.environ['feature_registry__purview__purview_name'] = f'{purview_name}'
    feathr_output_path = f'abfss://{adls_fs_name}@{adls_account}.dfs.core.windows.net/feathr_output'

    return feathr_output_path


def feathr_udf_day_calc(df: DataFrame) -> DataFrame:
    df = df.withColumn("fare_amount_cents", col("fare_amount")*100)
    return df


def build_data_source(data_path):
    batch_source = HdfsSource(name="nycTaxiBatchSource",
                              path=data_path,
                              event_timestamp_column="lpep_dropoff_datetime",
                              preprocessing=feathr_udf_day_calc,
                              timestamp_format="yyyy-MM-dd HH:mm:ss")

    return batch_source


def build_features(data_source):
    ##############################
    # define anchored feature
    ##############################
    f_trip_distance = Feature(name="f_trip_distance",
                              feature_type=FLOAT,
                              transform="trip_distance")

    f_trip_time_duration = Feature(name="f_trip_time_duration",
                                   feature_type=INT32,
                                   transform="(to_unix_timestamp(lpep_dropoff_datetime) - to_unix_timestamp(lpep_pickup_datetime))/60")

    f_is_long_trip_distance = Feature(name="f_is_long_trip_distance",
                                      feature_type=BOOLEAN,
                                      transform="cast_float(trip_distance)>30")

    f_day_of_week = Feature(name="f_day_of_week",
                            feature_type=INT32,
                            transform="dayofweek(lpep_dropoff_datetime)")

    features = [
        f_trip_distance,
        f_trip_time_duration,
        f_is_long_trip_distance,
        f_day_of_week
    ]

    request_anchor = FeatureAnchor(name="request_features",
                                   source=INPUT_CONTEXT,
                                   features=features)

    ################################################
    # define anchored (aggregated) feature
    ################################################
    location_id = TypedKey(key_column="DOLocationID",
                           key_column_type=ValueType.INT32,
                           description="location id in NYC",
                           full_name="nyc_taxi.location_id")

    f_location_avg_fare = Feature(name="f_location_avg_fare",
                                  key=location_id,
                                  feature_type=FLOAT,
                                  transform=WindowAggTransformation(agg_expr="cast_float(fare_amount)",
                                                                    agg_func="AVG",
                                                                    window="90d"))

    f_location_max_fare = Feature(name="f_location_max_fare",
                                  key=location_id,
                                  feature_type=FLOAT,
                                  transform=WindowAggTransformation(agg_expr="cast_float(fare_amount)",
                                                                    agg_func="MAX",
                                                                    window="90d"))

    f_location_total_fare_cents = Feature(name="f_location_total_fare_cents",
                                          key=location_id,
                                          feature_type=FLOAT,
                                          transform=WindowAggTransformation(agg_expr="fare_amount_cents",
                                                                            agg_func="SUM",
                                                                            window="90d"))

    agg_features = [
        f_location_avg_fare,
        f_location_max_fare,
        f_location_total_fare_cents
    ]

    agg_anchor = FeatureAnchor(name="aggregationFeatures",
                               source=data_source,
                               features=agg_features)

    ##############################
    # define DerivedFeature
    ##############################
    f_trip_time_distance = DerivedFeature(name="f_trip_time_distance",
                                          feature_type=FLOAT,
                                          input_features=[f_trip_distance, f_trip_time_duration],
                                          transform="f_trip_distance * f_trip_time_duration")

    f_trip_time_rounded = DerivedFeature(name="f_trip_time_rounded",
                                         feature_type=INT32,
                                         input_features=[f_trip_time_duration],
                                         transform="f_trip_time_duration % 10")

    anchored_feature_dict = dict()
    anchored_feature_dict["request_anchor"] = request_anchor
    anchored_feature_dict["agg_anchor"] = agg_anchor

    derived_feature_dict = dict()
    derived_feature_dict["f_trip_time_distance"] = f_trip_time_distance
    derived_feature_dict["f_trip_time_rounded"] = f_trip_time_rounded

    key_dict = dict()
    key_dict["location_id"] = location_id

    return anchored_feature_dict, derived_feature_dict, key_dict


def download_result_df(client: FeathrClient) -> pd.DataFrame:
    """Download the job result dataset from cloud as a Pandas dataframe."""
    res_url = client.get_job_result_uri(block=True, timeout_sec=600)
    tmp_dir = tempfile.TemporaryDirectory()
    client.feathr_spark_launcher.download_result(result_path=res_url, local_folder=tmp_dir.name)
    dataframe_list = []
    # assuming the result are in avro format
    for file in glob.glob(os.path.join(tmp_dir.name, '*.avro')):
        dataframe_list.append(pdx.read_avro(file))
    vertical_concat_df = pd.concat(dataframe_list, axis=0)
    tmp_dir.cleanup()
    return vertical_concat_df


def dataset_preparation(dataframe_result):
    final_df = dataframe_result
    final_df.drop(["lpep_pickup_datetime", "lpep_dropoff_datetime",
                   "store_and_fwd_flag"], axis=1, inplace=True, errors='ignore')
    final_df.fillna(0, inplace=True)
    final_df['fare_amount'] = final_df['fare_amount'].astype("float64")

    train_x, test_x, train_y, test_y = train_test_split(final_df.drop(["fare_amount"], axis=1),
                                                        final_df["fare_amount"],
                                                        test_size=0.2,
                                                        random_state=42)

    return train_x, test_x, train_y, test_y


def main():
    feathr_output = config_feathr()
    feathr_runtime_config = config_runtime()

    # create feathr client
    client = FeathrClient(config_path=feathr_runtime_config.name, local_workspace_dir="/Users/ruiliu/Develop/tmp")

    # build data source
    wasbs_path = "wasbs://public@azurefeathrstorage.blob.core.windows.net/sample_data/green_tripdata_2020-04_with_index.csv"
    batch_source = build_data_source(wasbs_path)

    anchored_feature_dict, derived_feature_dict, key_dict = build_features(batch_source)

    # build features
    client.build_features(anchor_list=[anchored_feature_dict["agg_anchor"],
                                       anchored_feature_dict["request_anchor"]],
                          derived_feature_list=[derived_feature_dict["f_trip_time_distance"],
                                                derived_feature_dict["f_trip_time_rounded"]])

    # config output path
    if client.spark_runtime == 'databricks':
        output_path = 'dbfs:/feathrazure_test.avro'
    else:
        output_path = feathr_output

    feature_query = FeatureQuery(feature_list=["f_location_avg_fare",
                                               "f_trip_time_rounded",
                                               "f_is_long_trip_distance",
                                               "f_location_total_fare_cents"],
                                 key=key_dict["location_id"])

    settings = ObservationSettings(observation_path=wasbs_path,
                                   event_timestamp_column="lpep_dropoff_datetime",
                                   timestamp_format="yyyy-MM-dd HH:mm:ss")
    start = time.perf_counter()
    client.get_offline_features(observation_settings=settings,
                                feature_query=feature_query,
                                output_path=output_path)
    client.wait_job_to_finish(timeout_sec=500)
    end = time.perf_counter()
    print("#### Time of building feature from offline store: {} seconds ####".format(end - start))
    perf_description = ("#### Feature building includes extracting data from offline store [Windows Azure Storage Blob]"
                        ", feature computation, putting back to offline store [Azure Blob File System] ####")
    print(perf_description)

    start = time.perf_counter()
    df_res = download_result_df(client)
    end = time.perf_counter()
    print("#### Time of downloading data from offline store to local: {} seconds ####".format(end - start))
    print("Results: {}".format(df_res))

    # build dataset for training and serving
    train_x, test_x, train_y, test_y = dataset_preparation(df_res)

    # model training and serving
    model = GradientBoostingRegressor()
    model.fit(train_x, train_y)
    y_predict = model.predict(test_x)
    y_actual = test_y.values.flatten().tolist()

    sum_actuals = sum_errors = 0

    for actual_val, predict_val in zip(y_actual, y_predict):
        abs_error = actual_val - predict_val
        if abs_error < 0:
            abs_error = abs_error * -1

        sum_errors = sum_errors + abs_error
        sum_actuals = sum_actuals + actual_val

    mean_abs_percent_error = sum_errors / sum_actuals
    print("Model MAPE: {}".format(mean_abs_percent_error))
    print("Model Accuracy: {}".format(1 - mean_abs_percent_error))

    # define backfill time
    backfill_time = BackfillTime(start=datetime(2020, 5, 20), end=datetime(2020, 5, 20), step=timedelta(days=1))

    ########################################################
    # Materialize feature value into online storage
    ########################################################
    redisSink = RedisSink(table_name="nycTaxiDemoFeature")
    settings = MaterializationSettings("nycTaxiTable",
                                       backfill_time=backfill_time,
                                       sinks=[redisSink],
                                       feature_names=["f_location_avg_fare", "f_location_max_fare"])

    start = time.perf_counter()
    client.materialize_features(settings)
    client.wait_job_to_finish(timeout_sec=500)
    end = time.perf_counter()
    print("#### Time of materializing features to online store: {} seconds ####".format(end - start))

    # Fetching three features from online store
    start = time.perf_counter()
    multiple_res_online_store = client.multi_get_online_features(feature_table='nycTaxiDemoFeature',
                                                                 keys=["239", "248", "265"],
                                                                 feature_names=['f_location_avg_fare',
                                                                                'f_location_max_fare'])
    end = time.perf_counter()
    print("#### Time of fetching three features from online store after materialization: {} seconds ####".format(
        end - start))
    print("Feature from online store: {}".format(multiple_res_online_store))

    # Fetching three features from online store
    start = time.perf_counter()
    multiple_res_online_store = client.multi_get_online_features(feature_table='nycTaxiDemoFeature',
                                                                 keys=["123", "144", "11"],
                                                                 feature_names=['f_location_avg_fare',
                                                                                'f_location_max_fare'])
    end = time.perf_counter()
    print("#### Time of fetching three features from online store after materialization: {} seconds ####".format(
        end - start))
    print("Feature from online store: {}".format(multiple_res_online_store))

    # Fetching three features from online store
    start = time.perf_counter()
    multiple_res_online_store = client.multi_get_online_features(feature_table='nycTaxiDemoFeature',
                                                                 keys=["239", "248", "265"],
                                                                 feature_names=['f_location_avg_fare',
                                                                                'f_location_max_fare'])
    end = time.perf_counter()
    print("#### Time of fetching three features from online store after materialization: {} seconds ####".format(
        end - start))
    print("Feature from online store: {}".format(multiple_res_online_store))

    a = list(range(1, 265 + 1))
    a_str = list(map(str, a))
    a_str_set = set(a_str)
    b_str_set = {"103", "104", "105", "109", "110", "12", "172", "176", "199", "5", "84", "99", "187", "245", "44", "6"}
    feature_keys = list(a_str_set - b_str_set)

    # Fetching all features from online store
    start = time.perf_counter()
    multiple_res_online_store = client.multi_get_online_features(feature_table='nycTaxiDemoFeature',
                                                                 keys=feature_keys,
                                                                 feature_names=['f_location_avg_fare',
                                                                                'f_location_max_fare'])
    end = time.perf_counter()
    print("#### Time of fetching all features from online store after materialization: {} seconds ####".format(
        end - start))
    print("Feature from online store: {}".format(multiple_res_online_store))

    ########################################################
    # Materialize feature value into offline storage
    ########################################################
    offline_sink = HdfsSink(output_path=output_path)
    settings = MaterializationSettings("nycTaxiTable",
                                       backfill_time=backfill_time,
                                       sinks=[offline_sink],
                                       feature_names=["f_location_avg_fare", "f_location_max_fare"])

    start = time.perf_counter()
    client.materialize_features(settings)
    client.wait_job_to_finish(timeout_sec=900)
    end = time.perf_counter()
    print("#### Time of materializing features to offline store: {} seconds ####".format(end - start))

    # Downloading feature value from offline store
    start = time.perf_counter()
    res_offline_store = get_result_df(client, "avro", output_path + "/df0/daily/2020/05/20")
    end = time.perf_counter()
    print("#### Time of fetching all features from offline store after materializing: {} seconds ####".format(
        end - start))
    print("Feature from offline store:")
    # pd.set_option('display.max_rows', None)
    print(res_offline_store)

    # Downloading feature value from offline store
    start = time.perf_counter()
    res_offline_store = get_result_df(client, "avro", output_path + "/df0/daily/2020/05/20")
    end = time.perf_counter()
    print("#### Time of fetching all features from offline store after materializing: {} seconds ####".format(
        end - start))
    print("Feature from offline store:")
    # pd.set_option('display.max_rows', None)
    print(res_offline_store)


if __name__ == "__main__":
    main()
