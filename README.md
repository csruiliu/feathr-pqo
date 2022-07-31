# Feathr-PQO

Feathr-PQO is built on top of [feathr](https://github.com/linkedin/feathr) v0.6.0.

## Installation

### Prerequisite

Python 3.8.12

SBT: 1.6.2

Apache Spark: 3.1.3

Scala: 2.12.15

### Build Feathr Python 

Feathr assumes that we have the Apache Kafka C/C++ client library installed. 

To install it, follow the [install instructions](https://github.com/edenhill/librdkafka#installation) on the librdkafka home page.

```
cd /feathr-pqo/feathr_project (where setup.py is located)
pip3 install feathr
```

### Build Feathr Scala  

Download and Configure SBT

`sonatype.sbt` is not necessary if you want to test locally.

```
sbt compile
```






