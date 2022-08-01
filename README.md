# Feathr-PQO

Feathr-PQO is built on top of [Feathr](https://github.com/linkedin/feathr) v0.6.0. 

Feathr is the feature store that is used in production in LinkedIn for many years and was open sourced in April 2022.


# Installation

## Prerequisite

Python 3.8.12

SBT: 1.6.2

Apache Spark: 3.1.3

Scala: 2.12.15

## Build Feathr Python 

Feathr assumes that we have the Apache Kafka C/C++ client library installed. 

To install it, follow the [install instructions](https://github.com/edenhill/librdkafka#installation) on the librdkafka home page.

Then, we can install feathr  python library 

```
cd /feathr-pqo/feathr_project (where setup.py is located)
pip3 install feathr
```

## Build Feathr Scala  

Download and Configure SBT

`sonatype.sbt` is not necessary if you don't need to publish to private cloud repo.

Compile Project using SBT

```
cd feathr
sbt compile
```

Check `feathr/project/plugins.sbt` to see if `sbt-assembly` is configured, then run the assembly command to generate the Jar file 

```
# make sure in the feathr directory
sbt assembly [or sbt clean assembly] 
```

The generated Jar file should be in `feathr/target/scala-XXX`


