import os
import glob
import tempfile
import pandas as pd
import pandavro as pdx

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
from feathr.utils.job_utils import get_result_df

from feathr.utils._envvariableutil import _EnvVaraibleUtil


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
    # Explanations on the meaning of each field can be found at
    # https://github.com/linkedin/feathr/blob/main/feathr_project/feathrcli/data/feathr_user_workspace/feathr_config.yaml

    project_config = """
        api_version: 1
        project_config:
          project_name: 'feathr_getting_started'
          required_environment_variables:
            - 'REDIS_PASSWORD'
            - 'AZURE_CLIENT_ID'
            - 'AZURE_TENANT_ID'
            - 'AZURE_CLIENT_SECRET'
        """

    spark_config = """
        spark_config:
          spark_cluster: 'azure_synapse'
          spark_result_output_parts: '1'
          azure_synapse:
            dev_url: 'https://feathrazuretest3synapse.dev.azuresynapse.net'
            pool_name: 'spark3'
            workspace_dir: 'abfss://feathrazuretest3fs@feathrazuretest3storage.dfs.core.windows.net/feathr_getting_started'
            executor_size: 'Small'
            executor_num: 4
            feathr_runtime_location: https://feathrpqoplusdls.blob.core.windows.net/feathrpqoplusfs/feathr_project/feathr-assembly-0.6.0.jar
            feathr_runtime_location_sas: sp=r&st=2022-08-01T21:17:54Z&se=2022-08-31T05:17:54Z&spr=https&sv=2021-06-08&sr=b&sig=gNhx0cXtpSpNSYmzLjfRhKjrNQ1BDmKyTEozn%2BKr8c4%3D
          databricks:
            workspace_instance_url: 'https://adb-2474129336842816.16.azuredatabricks.net'
            config_template: {'run_name':'','new_cluster':{'spark_version':'9.1.x-scala2.12','node_type_id':'Standard_D3_v2','num_workers':2,'spark_conf':{}},'libraries':[{'jar':''}],'spark_jar_task':{'main_class_name':'','parameters':['']}}
            work_dir: 'dbfs:/feathr_getting_started'
            feathr_runtime_location: https://azurefeathrstorage.blob.core.windows.net/public/feathr-assembly-LATEST.jar
        """

    offline_store_config = """
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
        """

    online_store_config = """
        online_store:
          redis:
            host: 'feathrazuretest3redis.redis.cache.windows.net'
            port: 6380
            ssl_enabled: True
        """

    feature_registry_config = """
        feature_registry:
          purview:
            type_system_initialization: true
            purview_name: 'feathrazuretest3-purview1'
            delimiter: '__'
        """

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    with open(tmp.name, "w") as text_file:
        text_file.write(project_config)
        text_file.write(spark_config)
        text_file.write(offline_store_config)
        text_file.write(online_store_config)
        text_file.write(feature_registry_config)
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


def main():
    print("########################### \n## Config Feathr \n###########################")
    feathr_output = config_credentials()
    feathr_runtime_config = config_runtime()
    
    # create feathr client
    client = FeathrClient(config_path=feathr_runtime_config.name, local_workspace_dir="/Users/ruiliu/Develop/tmp")

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

    ####################################################
    # define anchored features using user profile table
    ####################################################
    # define user_id key
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

    #################################
    # build features using client
    #################################
    client.build_features(anchor_list=[anchored_features, agg_anchored_features])

    ################################################
    # create training data using point-in-time join
    ################################################
    if client.spark_runtime == 'databricks':
        output_path = 'dbfs:/feathrazure_test.avro'
    else:
        output_path = feathr_output

    print("Output Path: {}".format(output_path))

    feature_query = FeatureQuery(feature_list=["feature_user_age",
                                               "feature_user_tax_rate",
                                               "feature_user_gift_card_balance",
                                               "feature_user_has_valid_credit_card",
                                               "feature_user_total_purchase_in_90days"],
                                 key=user_id)

    settings = ObservationSettings(observation_path=user_observation_wasbs,
                                   event_timestamp_column="event_timestamp",
                                   timestamp_format="yyyy-MM-dd")

    client.get_offline_features(observation_settings=settings,
                                feature_query=feature_query,
                                output_path=output_path)

    client.wait_job_to_finish(timeout_sec=1000)


if __name__ == "__main__":
    main()
