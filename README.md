# Unit testing for checking the functionality of the cluster

To build the  Docker Image:

```shell
docker build . -t test_smile
```

To run the built image:

```shell
docker run --rm --gpus 0  --memory="32g" --shm-size 2g test_smile
```
It should download some data, and train a model for 2 epochs which usually takes around 2-3 minutes.
