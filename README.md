# NexusMiner



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
```

  ./NexusMiner ../../myownminer.conf

## DEPENDENCIES

### General


### Windows (Not yet supported)

* MPIR: Windows GMP equivalent

### Ubuntu/Debian

* GMP:          sudo apt-get install libgmp3-dev
* OpenSSL:      sudo apt-get install libssl-dev
