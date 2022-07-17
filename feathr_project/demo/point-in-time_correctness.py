import os
import glob
import tempfile
from math import sqrt
import pandas as pd
import pandavro as pdx
from datetime import datetime, timedelta

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from feathr import FeathrClient, FeatureQuery, ObservationSettings
from feathr import TypedKey
from feathr import BOOLEAN, FLOAT, INT32, ValueType
from feathr import Feature, DerivedFeature, FeatureAnchor
from feathr import HdfsSource
from feathr import WindowAggTransformation
from feathr import RedisSink, HdfsSink
from feathr import BackfillTime, MaterializationSettings
from feathr.job_utils import get_result_df


def config_credentials():
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


def config_runtime():
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


def feathr_udf_preprocessing(df: DataFrame) -> DataFrame:
    df = df.withColumn("tax_rate_decimal", col("tax_rate")/100)
    df.show(10)
    return df


def show_all_datasets():
    # path of observation dataset (aka label dataset)
    user_observation_afs = ("https://azurefeathrstorage.blob.core.windows.net/"
                            "public/sample_data/product_recommendation_sample/"
                            "user_observation_mock_data.csv")

    # path of user profile dataset (dataset used to generate user features)
    user_profile_afs = ("https://azurefeathrstorage.blob.core.windows.net/"
                        "public/sample_data/product_recommendation_sample/"
                        "user_profile_mock_data.csv")

    # path of purchase history dataset (dataset used to generate user features)
    # This is activity data, so we need to use aggregation to generation features
    user_purchase_history_afs = ("https://azurefeathrstorage.blob.core.windows.net/"
                                 "public/sample_data/product_recommendation_sample/"
                                 "user_purchase_history_mock_data.csv")

    user_observation_mock_data = pd.read_csv(user_observation_afs)
    user_profile_mock_data = pd.read_csv(user_profile_afs)
    user_purchase_history_mock_data = pd.read_csv(user_purchase_history_afs)

    # save dataset to local csv files
    user_observation_mock_data.to_csv("user_observation_mock_data.csv")
    user_profile_mock_data.to_csv("user_profile_mock_data.csv")
    user_purchase_history_mock_data.to_csv("user_purchase_history_mock_data.csv")

    print("### User Observation Dataset ###")
    print(user_observation_afs)

    print("### User Profile Dataset ###")
    print(user_profile_afs)

    print("### User Purchase_history ###")
    print(user_purchase_history_afs)


def main():
    print("########################### \n## Config Feathr \n###########################")
    feathr_output = config_credentials()
    feathr_runtime_config = config_runtime()

    # create feathr client
    client = FeathrClient(config_path=feathr_runtime_config.name, local_workspace_dir="/Users/ruiliu/Develop/tmp")

    # show all the dataset and save them to local csv files
    # show_all_datasets()

    ####################
    # datasets path
    ####################
    user_observation_wasbs = ("wasbs://public@azurefeathrstorage.blob.core.windows.net/"
                              "sample_data/product_recommendation_sample/"
                              "user_observation_mock_data.csv")

    user_profile_wasbs = ("wasbs://public@azurefeathrstorage.blob.core.windows.net/"
                          "sample_data/product_recommendation_sample/"
                          "user_profile_mock_data.csv")

    user_purchase_history_wasbs = ("wasbs://public@azurefeathrstorage.blob.core.windows.net/"
                                   "sample_data/product_recommendation_sample/"
                                   "user_purchase_history_mock_data.csv")

    # define primary key
    user_id = TypedKey(key_column="user_id",
                       key_column_type=ValueType.INT32,
                       description="user id",
                       full_name="product_recommendation.user_id")

    ######################
    # define anchored key
    ######################
    feature_user_age = Feature(name="feature_user_age",
                               key=user_id,
                               feature_type=INT32, transform="age")

    feature_user_tax_rate = Feature(name="feature_user_tax_rate",
                                    key=user_id,
                                    feature_type=FLOAT,
                                    transform="tax_rate_decimal")

    feature_user_gift_card_balance = Feature(name="feature_user_gift_card_balance",
                                             key=user_id,
                                             feature_type=FLOAT,
                                             transform="gift_card_balance")

    feature_user_has_valid_credit_card = Feature(name="feature_user_has_valid_credit_card",
                                                 key=user_id,
                                                 feature_type=BOOLEAN,
                                                 transform="number_of_credit_cards > 0")

    user_profile_source = HdfsSource(name="user_profile_data",
                                     path=user_profile_wasbs,
                                     preprocessing=feathr_udf_preprocessing)

    anchored_features = FeatureAnchor(name="anchored_features",
                                      source=user_profile_source,
                                      features=[feature_user_age,
                                                feature_user_tax_rate,
                                                feature_user_gift_card_balance,
                                                feature_user_has_valid_credit_card])

    #################################
    # define aggregated anchored key
    #################################
    # this time window is based on the event_timestamp_column in observation dataset
    # i.e., time travel back "window" period from the time of event_timestamp_column
    agg_feature_user_purchase_90d = Feature(name="feature_user_total_purchase_in_90days",
                                            key=user_id,
                                            feature_type=FLOAT,
                                            transform=WindowAggTransformation(agg_expr="cast_float(purchase_amount)",
                                                                              agg_func="SUM",
                                                                              window="90d"))

    # we need event_timestamp_column since we need to decide if the purchase happens within the "time window"
    purchase_history_source = HdfsSource(name="purchase_history_data",
                                         path=user_purchase_history_wasbs,
                                         event_timestamp_column="purchase_date",
                                         timestamp_format="yyyy-MM-dd")

    agg_anchored_features = FeatureAnchor(name="aggregation_anchored_features",
                                          source=purchase_history_source,
                                          features=[agg_feature_user_purchase_90d])

    #####################
    # define derived key
    #####################

    feature_user_purchasing_power = DerivedFeature(name="feature_user_purchasing_power",
                                                   key=user_id,
                                                   feature_type=FLOAT,
                                                   input_features=[feature_user_gift_card_balance,
                                                                   feature_user_has_valid_credit_card],
                                                   transform="feature_user_gift_card_balance + if_else(toBoolean(feature_user_has_valid_credit_card), 100, 0)")

    # build features in client
    client.build_features(anchor_list=[anchored_features, agg_anchored_features],
                          derived_feature_list=[feature_user_purchasing_power])

    if client.spark_runtime == 'databricks':
        output_path = 'dbfs:/feathrazure_test.avro'
    else:
        output_path = feathr_output

    feature_query = FeatureQuery(feature_list=["feature_user_age",
                                               "feature_user_tax_rate",
                                               "feature_user_gift_card_balance",
                                               "feature_user_has_valid_credit_card",
                                               "feature_user_total_purchase_in_90days",
                                               "feature_user_purchasing_power"],
                                 key=user_id)

    settings = ObservationSettings(observation_path=user_observation_wasbs,
                                   event_timestamp_column="event_timestamp",
                                   timestamp_format="yyyy-MM-dd")

    client.get_offline_features(observation_settings=settings,
                                feature_query=feature_query,
                                output_path=output_path)
    client.wait_job_to_finish(timeout_sec=500)

    df_res = download_result_df(client)
    # print("Generated Features from Offline Store:")
    # print(df_res)
    # df_res.to_csv("join_all.csv")

    #################################
    # Train a machine learning model
    #################################
    final_df = df_res
    final_df.drop(["event_timestamp"], axis=1, inplace=True, errors='ignore')
    final_df.fillna(0, inplace=True)
    final_df['product_rating'] = final_df['product_rating'].astype("float64")

    final_df.to_csv("train_dataset.csv")

    train_x, test_x, train_y, test_y = train_test_split(final_df.drop(["product_rating"], axis=1),
                                                        final_df["product_rating"],
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
    print("Model MAPE: \n{}".format(mean_abs_percent_error))
    print("Model Accuracy: \n{}".format(1 - mean_abs_percent_error))

    """
    #################################
    # Online Materialization
    #################################
    backfill_time = BackfillTime(start=datetime(2020, 5, 20), end=datetime(2020, 5, 23), step=timedelta(days=1))
    redisSink = RedisSink(table_name="productRecommendationDemoFeature")
    settings = MaterializationSettings(name="productRecommendationFeatureSetting",
                                       backfill_time=backfill_time,
                                       sinks=[redisSink],
                                       feature_names=["feature_user_age", "feature_user_purchasing_power"])

    client.materialize_features(settings)
    client.wait_job_to_finish(timeout_sec=500)

    results_online = client.get_online_features(feature_table='productRecommendationDemoFeature',
                                                key='2',
                                                feature_names=['feature_user_age', 'feature_user_purchasing_power'])

    print("=== Fetch features from online store: ===")
    print(results_online)
    """

    #################################
    # Offline Materialization
    #################################
    backfill_time = BackfillTime(start=datetime(2020, 5, 20), end=datetime(2020, 5, 22), step=timedelta(days=1))
    hdfsSink = HdfsSink(output_path=output_path)
    settings = MaterializationSettings(name="productRecommendationFeatureSetting",
                                       backfill_time=backfill_time,
                                       sinks=[hdfsSink],
                                       feature_names=["feature_user_age", "feature_user_purchasing_power"])

    client.materialize_features(settings)
    client.wait_job_to_finish(timeout_sec=900)

    res_offline_store_0520 = get_result_df(client, "avro", output_path + "/df0/daily/2020/05/20")
    print("=== Fetch features from online store: ===")
    for key, value in res_offline_store_0520.items():
        print("{}: {}".format(key, value))

    res_offline_store_0521 = get_result_df(client, "avro", output_path + "/df0/daily/2020/05/21")
    print("=== Fetch features from online store: ===")
    for key, value in res_offline_store_0521.items():
        print("{}: {}".format(key, value))

    res_offline_store_0522 = get_result_df(client, "avro", output_path + "/df0/daily/2020/05/22")
    print("=== Fetch features from online store: ===")
    for key, value in res_offline_store_0522.items():
        print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
