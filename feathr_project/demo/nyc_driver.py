import glob
import os
import tempfile
from datetime import datetime, timedelta
from math import sqrt

import pandas as pd
import pandavro as pdx
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import dayofweek, dayofyear, col
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from feathr import FeathrClient
from feathr import BOOLEAN, FLOAT, INT32, ValueType
from feathr import Feature, DerivedFeature, FeatureAnchor
from feathr import BackfillTime, MaterializationSettings
from feathr import FeatureQuery, ObservationSettings
from feathr import RedisSink
from feathr import INPUT_CONTEXT, HdfsSource
from feathr import WindowAggTransformation
from feathr import TypedKey


def config_environment():
    import tempfile
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


def feathr_udf_day_calc(df: DataFrame) -> DataFrame:
    df = df.withColumn("fare_amount_cents", col("fare_amount")*100)
    return df


def get_result_df(client: FeathrClient) -> pd.DataFrame:
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


def main():
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
    os.environ['spark_config__azure_synapse__workspace_dir'] = f'abfss://{adls_fs_name}@{adls_account}.dfs.core.windows.net/feathr_project'
    os.environ['feature_registry__purview__purview_name'] = f'{purview_name}'
    os.environ['online_store__redis__host'] = redis_host
    os.environ['online_store__redis__port'] = redis_port
    os.environ['online_store__redis__ssl_enabled'] = redis_ssl
    os.environ['REDIS_PASSWORD'] = redis_password
    os.environ['feature_registry__purview__purview_name'] = f'{purview_name}'
    feathr_output_path = f'abfss://{adls_fs_name}@{adls_account}.dfs.core.windows.net/feathr_output'

    feathr_tmp = config_environment()

    #######################
    # Create Feathr Client
    #######################
    client = FeathrClient(config_path=feathr_tmp.name, local_workspace_dir="/Users/ruiliu/Develop/tmp")
    dpath = "https://azurefeathrstorage.blob.core.windows.net/public/sample_data/green_tripdata_2020-04_with_index.csv"
    pd.read_csv(dpath)

    wasbs_path = "wasbs://public@azurefeathrstorage.blob.core.windows.net/sample_data/green_tripdata_2020-04_with_index.csv"
    batch_source = HdfsSource(name="nycTaxiBatchSource",
                              path=wasbs_path,
                              event_timestamp_column="lpep_dropoff_datetime",
                              preprocessing=feathr_udf_day_calc,
                              timestamp_format="yyyy-MM-dd HH:mm:ss")

    ##############################
    # Define Anchors and Features
    ##############################
    f_trip_distance = Feature(name="f_trip_distance",
                              feature_type=FLOAT, transform="trip_distance")
    f_trip_time_duration = Feature(name="f_trip_time_duration",
                                   feature_type=INT32,
                                   transform="(to_unix_timestamp(lpep_dropoff_datetime) - to_unix_timestamp(lpep_pickup_datetime))/60")

    features = [
        f_trip_distance,
        f_trip_time_duration,
        Feature(name="f_is_long_trip_distance",
                feature_type=BOOLEAN,
                transform="cast_float(trip_distance)>30"),
        Feature(name="f_day_of_week",
                feature_type=INT32,
                transform="dayofweek(lpep_dropoff_datetime)"),
    ]

    request_anchor = FeatureAnchor(name="request_features",
                                   source=INPUT_CONTEXT,
                                   features=features)

    ##############################
    # Window aggregation features
    ##############################
    location_id = TypedKey(key_column="DOLocationID",
                           key_column_type=ValueType.INT32,
                           description="location id in NYC",
                           full_name="nyc_taxi.location_id")
    agg_features = [Feature(name="f_location_avg_fare",
                            key=location_id,
                            feature_type=FLOAT,
                            transform=WindowAggTransformation(agg_expr="cast_float(fare_amount)",
                                                              agg_func="AVG",
                                                              window="90d")),
                    Feature(name="f_location_max_fare",
                            key=location_id,
                            feature_type=FLOAT,
                            transform=WindowAggTransformation(agg_expr="cast_float(fare_amount)",
                                                              agg_func="MAX",
                                                              window="90d")),
                    Feature(name="f_location_total_fare_cents",
                            key=location_id,
                            feature_type=FLOAT,
                            transform=WindowAggTransformation(agg_expr="fare_amount_cents",
                                                              agg_func="SUM",
                                                              window="90d")),
                    ]

    agg_anchor = FeatureAnchor(name="aggregationFeatures",
                               source=batch_source,
                               features=agg_features)

    ###########################
    # Derived Features Section
    ###########################
    f_trip_time_distance = DerivedFeature(name="f_trip_time_distance",
                                          feature_type=FLOAT,
                                          input_features=[f_trip_distance, f_trip_time_duration],
                                          transform="f_trip_distance * f_trip_time_duration")

    f_trip_time_rounded = DerivedFeature(name="f_trip_time_rounded",
                                         feature_type=INT32,
                                         input_features=[f_trip_time_duration],
                                         transform="f_trip_time_duration % 10")

    ###########################
    # Employ Features
    ###########################
    client.build_features(anchor_list=[agg_anchor, request_anchor],
                          derived_feature_list=[f_trip_time_distance, f_trip_time_rounded])

    ################################################################
    # Create training data using point-in-time correct feature join
    ################################################################
    if client.spark_runtime == 'databricks':
        output_path = 'dbfs:/feathrazure_test.avro'
    else:
        output_path = feathr_output_path

    feature_query = FeatureQuery(feature_list=["f_location_avg_fare",
                                               "f_trip_time_rounded",
                                               "f_is_long_trip_distance",
                                               "f_location_total_fare_cents"],
                                 key=location_id)
    settings = ObservationSettings(observation_path=wasbs_path,
                                   event_timestamp_column="lpep_dropoff_datetime",
                                   timestamp_format="yyyy-MM-dd HH:mm:ss")
    client.get_offline_features(observation_settings=settings,
                                feature_query=feature_query,
                                output_path=output_path)
    client.wait_job_to_finish(timeout_sec=500)

    ##########################################
    # Download the result and show the result
    ##########################################
    df_res = get_result_df(client)
    print("Results: {}".format(df_res))

    #################################
    # Train a machine learning model
    #################################

    final_df = df_res
    final_df.drop(["lpep_pickup_datetime", "lpep_dropoff_datetime",
                   "store_and_fwd_flag"], axis=1, inplace=True, errors='ignore')
    final_df.fillna(0, inplace=True)
    final_df['fare_amount'] = final_df['fare_amount'].astype("float64")

    train_x, test_x, train_y, test_y = train_test_split(final_df.drop(["fare_amount"], axis=1),
                                                        final_df["fare_amount"],
                                                        test_size=0.2,
                                                        random_state=42)
    model = GradientBoostingRegressor()
    model.fit(train_x, train_y)

    y_predict = model.predict(test_x)

    y_actual = test_y.values.flatten().tolist()
    rmse = sqrt(mean_squared_error(y_actual, y_predict))

    sum_actuals = sum_errors = 0

    for actual_val, predict_val in zip(y_actual, y_predict):
        abs_error = actual_val - predict_val
        if abs_error < 0:
            abs_error = abs_error * -1

        sum_errors = sum_errors + abs_error
        sum_actuals = sum_actuals + actual_val

    mean_abs_percent_error = sum_errors / sum_actuals
    print("Model MAPE:")
    print(mean_abs_percent_error)
    print()
    print("Model Accuracy:")
    print(1 - mean_abs_percent_error)

    ########################################################
    # Materialize feature value into offline/online storage
    ########################################################
    backfill_time = BackfillTime(start=datetime(2020, 5, 20), end=datetime(2020, 5, 20), step=timedelta(days=1))
    redisSink = RedisSink(table_name="nycTaxiDemoFeature")
    settings = MaterializationSettings("nycTaxiTable",
                                       backfill_time=backfill_time,
                                       sinks=[redisSink],
                                       feature_names=["f_location_avg_fare", "f_location_max_fare"])

    client.materialize_features(settings)
    client.wait_job_to_finish(timeout_sec=500)

    ##############################################
    # Fetching feature value for online inference
    ##############################################
    res = client.get_online_features('nycTaxiDemoFeature', '265', ['f_location_avg_fare', 'f_location_max_fare'])

    client.multi_get_online_features("nycTaxiDemoFeature",
                                     ["239", "265"],
                                     ['f_location_avg_fare', 'f_location_max_fare'])

    ####################################
    # Registering and Fetching features
    ####################################
    client.register_features()
    client.list_registered_features(project_name="feathr_getting_started")


if __name__ == "__main__":
    main()
