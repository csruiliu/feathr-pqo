import os

import pandas as pd

from feathr import FeathrClient

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


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


def main():
    print("########################### \n## Config Feathr \n###########################")
    feathr_output = config_credentials()
    feathr_runtime_config = config_runtime()

    # create feathr client
    client = FeathrClient(config_path=feathr_runtime_config.name, local_workspace_dir="/Users/ruiliu/Develop/tmp")

    # path of observation dataset (aka label dataset)
    user_observation_mock_data_path = ("https://azurefeathrstorage.blob.core.windows.net/"
                                       "public/sample_data/product_recommendation_sample/"
                                       "user_observation_mock_data.csv")

    # path of user profile dataset (dataset used to generate user features)
    user_profile_mock_data_path = ("https://azurefeathrstorage.blob.core.windows.net/"
                                   "public/sample_data/product_recommendation_sample/"
                                   "user_profile_mock_data.csv")

    # path of purchase history dataset (dataset used to generate user features)
    # This is activity data, so we need to use aggregation to generation features
    user_purchase_history_mock_data_path = ("https://azurefeathrstorage.blob.core.windows.net/"
                                            "public/sample_data/product_recommendation_sample/"
                                            "user_purchase_history_mock_data.csv")

    user_observation_mock_data = pd.read_csv(user_observation_mock_data_path)
    user_profile_mock_data = pd.read_csv(user_profile_mock_data_path)
    user_purchase_history_mock_data = pd.read_csv(user_purchase_history_mock_data_path)

    user_observation_mock_data.to_csv("user_observation_mock_data.csv")
    user_profile_mock_data.to_csv("user_profile_mock_data.csv")
    user_purchase_history_mock_data.to_csv("user_profile_mock_data.csv")
    

if __name__ == "__main__":
    main()