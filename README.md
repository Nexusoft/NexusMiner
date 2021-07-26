# NexusMiner

Miner for Nexus Hash channel with FPGA and (optional) CUDA GPU. Pool (currently hashpool.com, pool.blackminer.com) or Solo mining.  


## Wallet Setup

Ensure you are on latest wallet daemon release 5.0.x or greater. Ensure wallet has been unlocked for mining.

```
    -llpallowip=<ip-port>   ex: -llpallowip=192.168.0.1:9325 
                            note: this is not needed if you mine to localhost (127.0.0.1). This is primarily used for a local-area-network setup

    -mining                 Ensure mining LLP servers are initialized.
```



## COMMAND LINE OPTION ARGUMENTS

```
    <miner_config_file> Default=miner.conf
    -c --check          run config file check before miner startup
    -v --version        Show NexusMiner version
```

  ./NexusMiner ../../myownminer.conf -c

## BUILDING

Optional cmake build options are
* WITH_GPU_CUDA       to enable HASH channel gpu mining. CUDA Toolkit required
* WITH_PRIME          to enable PRIME channel mining (currently CPU only). primesieve required

### Windows
* [primesieve](https://github.com/kimwalisch/primesieve) (required for WITH_PRIME): 
    * Follow detailed [build instructions](https://github.com/kimwalisch/primesieve/blob/master/doc/BUILD.md) for Windows/Microsoft Visual C++.
    * You may need to add the location of primesieve.dll to your system path.  Try C:\Program Files (x86)\primesieve\bin
* OpenSSL: 
    * Download and run OpenSSL [installer](https://slproweb.com/products/Win32OpenSSL.html)

### Ubuntu/Debian

* primesieve (required for WITH_PRIME):  
    * sudo apt-get install libprimesieve-dev
* OpenSSL:
    * sudo apt-get install libssl-dev
