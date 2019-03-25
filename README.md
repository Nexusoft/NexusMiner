# NexusMiner

This is a Nexus Miner for the Prime and Hash Proof-of-Work Channels built from the ground up using the Nexus LLL-TAO Framework. It can target Solo or Pool Mining. It Supports CUDA and CPU and is easily extendible for additional hardware.


## CONFIGURATION SETTINGS

### config.ini

change GPU settings regarding seiving, testing

* nSievePrimesLog2
    ** How many sieving primes log base 2     (ex: 2^20 = 1048576 sieving primes)
* nSieveBitsLog2
    ** How large the sieving array log base 2 (ex: 2^23 = 8388608 sieve bits)

* nSieveIterationsLog2
    ** How many bit arrays should be seived before testing base 2 (ex: 2^10 = 1024 iterations)

* nTestLevels
    ** How many chains deep GPU test should go before passing workload to CPU
       (recommended to not test too deep, or CPU won't be saturated with enough work)


### offsets.ini

Change sieve offsets and primorial. You probably don't want to
change this unless you understand what is happening or want to experiment. see link
in file for more info on primes.

## COMMAND LINE OPTION ARGUMENTS

```
    -ip=<ip-address>    (default=127.0.0.1)
    -port=<port-number> (default=9325)
    -timeout=<timeout>  (default=10)
    -prime=<indices>    (i.e 0,1,2,3,4,5)
    -hash=<indices>     (i.e 0,1,2,3,4,5)
    -cpu                (tells miner to use cpu)
    -testnet            (specifies -port=8325)
```

  ./nexusminer -ip=192.168.0.100 -port=9325 -timeout=10 -prime=0,1,2 -hash=3,4,5

## DEPENDENCIES

### General

* CUDA Toolkit: follow installation guide: https://docs.nvidia.com/cuda/index.html

### Windows

* MPIR: Windows GMP equivalent

### Ubuntu/Debian

* GMP:          sudo apt-get install libgmp3-dev
* OpenSSL:      sudo apt-get install libssl-dev
