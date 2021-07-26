# NexusMiner

Miner for Nexus Hash channel with FPGA and (optional) CUDA GPU. Both pool and solo mining is supported. Supported pools are [hashpool](https://hashpool.com/coins/NXS) and [blackminer](https://pool.blackminer.com/).

## FPGA Support
List of supported [FPGA boards](https://github.com/Nexusoft/NexusMiner/blob/v2.0/docs/fpga_support.md). 

## Prime Channel
The miner currently supports CPU prime mining, solo only.  CPU prime mining is useful for development and testnet work but is not competitive with GPU prime mining on main net.  
## Wallet Setup

Ensure you are on latest wallet daemon release 5.0.x or greater. Ensure wallet has been unlocked for mining.

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

## Building (Cmake) 
Optional cmake build options are
* WITH_GPU_CUDA       to enable HASH channel gpu mining. CUDA Toolkit required
* WITH_PRIME          to enable PRIME channel mining (currently CPU only). primesieve and boost required

### Windows
* OpenSSL: 
    * Download and run OpenSSL [installer](https://slproweb.com/products/Win32OpenSSL.html)
* [primesieve](https://github.com/kimwalisch/primesieve) (required for WITH_PRIME): 
    * Follow detailed [build instructions](https://github.com/kimwalisch/primesieve/blob/master/doc/BUILD.md) for Windows/Microsoft Visual C++.
    * You may need to add the location of primesieve.dll to your system path.  Try C:\Program Files (x86)\primesieve\bin
* [boost](https://www.boost.org/users/download/) (required for WITH_PRIME):
    * Download and extract to C:\boost

### Ubuntu/Debian
* OpenSSL:
    * sudo apt-get install libssl-dev
* primesieve (required for WITH_PRIME):  
    * sudo apt-get install libprimesieve-dev
* boost (required for WITH_PRIME):
    * sudo apt-get install libboost-all-dev
