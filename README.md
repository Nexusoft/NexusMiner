# NexusMiner



## Wallet Setup

Ensure you are on latest wallet daemon release 3.0.x or greater. Ensure wallet has been unlocked for mining.

```
    -llpallowip=<ip-port>   ex: -llpallowip=192.168.0.1:9325 
                            note: this is not needed if you mine to localhost (127.0.0.1). This is primarily used for a local-area-network setup

    -mining                 Ensure mining LLP servers are initialized.

    -primemod               Use this command line option to have wallet generate a more favorable prime proof to work off of, increases ratio slightly. 
                            note: this option does draw more CPU usage when prime mining, so make sure you are running on a CPU that can handle a large load if you have a lot of workers.
```



## COMMAND LINE OPTION ARGUMENTS

```
    -ip=<ip-address>    Default=127.0.0.1
    -port=<port-number> Default=9325
    -timeout=<timeout>  Default=10
    -prime=<indices>    ex: 0,1,2,3,4,5
    -hash=<indices>     ex: 0,1,2,3,4,5
    -testnet            Specifies or overrides -port=8325
    -primeorigins       Standalone CPU mode to compute prime origins based on primorial end prime, base offset, and prime offsets.
```

  ./nexusminer -ip=192.168.0.100 -port=9325 -timeout=10 -prime=0,1,2,3,4,5
  ./nexusminer -ip=192.168.0.100 -port=9325 -timeout=10 -hash=0,1,2,3,4,5
  ./nexusminer -ip=192.168.0.100 -port=9325 -timeout=10 -prime=0,1,2 -hash=3,4,5

## DEPENDENCIES

### General


### Windows (Not yet supported)

* MPIR: Windows GMP equivalent

### Ubuntu/Debian

* GMP:          sudo apt-get install libgmp3-dev
* OpenSSL:      sudo apt-get install libssl-dev
