# NexusMiner

This is a Nexus Miner for the Prime and Hash Proof-of-Work Channels built from the ground up using the Nexus LLL-TAO Framework. It can target Solo or Pool Mining. It Supports CUDA and CPU and is easily extendible for additional hardware.


## CONFIGURATION SETTINGS

### config.ini

change GPU settings regarding seiving, testing

* nSievePrimesLog2
    ** How many sieving primes log base 2 (ex: 2^20 = 1048576 sieving primes)

* nSieveBitsLog2
    ** How large the sieving array log base 2 (ex: 2^23 = 8388608 sieve bits)

* nSieveIterationsLog2
    ** How many bit arrays should be seived before testing log base 2 (ex: 2^10 = 1024 iterations)

* nMaxCandidatesLog2
    ** How large the candidate buffer is log base 2 (ex: 2^16 = 65536 candidates)

* nTestLevels
    ** How many chains deep GPU test should go before passing workload to CPU
       (recommended to not test too deep, or CPU won't be saturated with enough work)


### offsets.ini

Change sieve offsets and primorial. You probably don't want to
change this unless you understand what is happening or want to experiment. see link
in file for more info on primes.

### origins.ini

Provided file specifying good prime origin locations to search where each offset has at least one prime over a handful of tests.
User generatable with -primeorigins command line option


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

* CUDA Toolkit: follow installation guide: https://docs.nvidia.com/cuda/index.html

### Windows (Not yet supported)

* MPIR: Windows GMP equivalent

### Ubuntu/Debian

* GMP:          sudo apt-get install libgmp3-dev
* OpenSSL:      sudo apt-get install libssl-dev
