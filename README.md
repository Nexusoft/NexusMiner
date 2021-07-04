# NexusMiner

Miner for Nexus Hash channel with FPGA. Pool (currently hashpool.com, pool.blackminer.com) or Solo mining


## Wallet Setup

Ensure you are on latest wallet daemon release 3.0.x or greater. Ensure wallet has been unlocked for mining.

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

## DEPENDENCIES

### General


### Windows (Not yet supported)

* MPIR: Windows GMP equivalent

### Ubuntu/Debian

* GMP:          sudo apt-get install libgmp3-dev
* OpenSSL:      sudo apt-get install libssl-dev
