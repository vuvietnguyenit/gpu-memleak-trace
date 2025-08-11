# cuda-trace

## Build

```sh
# cd cuda/
# gcc -fPIC -shared cuda_api.c -o libgpudevice.so
# cd nvlm/
# gcc -fPIC -shared nvlm_api.c -o libnvlmstats.so
```