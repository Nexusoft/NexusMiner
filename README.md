# NexusMiner

Miner for Nexus Hash and Prime channels with Hash channel FPGA support and (optional) CUDA GPU support for both hash and prime. Both pool and solo mining is supported. Supported hash channel pools are [hashpool](https://hashpool.com/coins/NXS) and [blackminer](https://pool.blackminer.com/).

## FPGA Support
FPGA mining for the hash channel is supported.  List of supported [FPGA boards](https://github.com/Nexusoft/NexusMiner/blob/master/docs/fpga_support.md). 

## Prime Channel
The miner supports prime mining for CPU and GPU, solo and pool.  Supported GPUs are Nvidia GTX 1070 or newer.  RTX 30 series GPUs provide the best profitability on the prime channel.

## Prime Pool
The prime pool is now open for beta testing.  To use the prime pool, set the following address and port in miner.conf:
```
    "wallet_ip" : "154.16.159.126",
    "port" : 50000,
```

## Wallet Setup

For solo mining use the latest wallet daemon release 5.0.5 or greater and ensure the wallet has been unlocked for mining.

```
    -llpallowip=<ip-port>   ex: -llpallowip=192.168.0.1:9325 
                            note: this is not needed if you mine to localhost (127.0.0.1). This is primarily used for a local-area-network setup

    -mining                 Ensure mining LLP servers are initialized.
```



## Command line option arguments
```
    <miner_config_file> Default=miner.conf
    -c --check          run config file check before miner startup
    -v --version        Show NexusMiner version
```

  ./NexusMiner ../../myownminer.conf -c
## Prebuilt Binaries (Windows x64)
Download NexusMiner.exe from the [latest release](https://github.com/Nexusoft/NexusMiner/releases). 

## Building (Cmake) 
Optional cmake build options are
* WITH_GPU_CUDA       to enable HASH channel gpu mining. CUDA Toolkit required
* WITH_PRIME          to enable PRIME channel mining. GMP and boost required
* Example cmake command: cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GPU_CUDA=On -DWITH_PRIME=On ..

### Windows
* OpenSSL: 
    * Download and run OpenSSL [installer](https://slproweb.com/products/Win32OpenSSL.html)
* [MPIR](http://www.mpir.org/) (required for WITH_PRIME):
    * Download and build.  Copy gmp*.lib and mpir.lib to NexusMiner/libs
* [boost](https://www.boost.org/users/download/) (required for WITH_PRIME):
    * Download and extract to C:\boost
### Ubuntu/Debian
* OpenSSL:
    * sudo apt-get install libssl-dev
* gmp (required for WITH_PRIME):  
    * sudo apt-get install libgmp-dev
* boost (required for WITH_PRIME):
    * sudo apt-get install libboost-all-dev
