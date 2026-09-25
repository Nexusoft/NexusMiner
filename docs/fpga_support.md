# FPGA Bitstreams
FPGA bitstreams for mining the Nexus Hash channel are currently avaiable for Beta testing on the following devices. For access, please make a request in the Nexus mining [telegram channel](https://t.me/NexusMiners).
FPGA Board | Manufacturer | Hash Rate (MH/s) | Power Estimate (W)
---------- | ------------ | ---------------- | ------------------
FK33 | SQRL | 700 | tbd
[KCU105](https://www.xilinx.com/products/boards-and-kits/kcu105.html) | Xilinx | 188 | 34
[Cmod A7-35T](https://digilent.com/reference/programmable-logic/cmod-a7/start) | Digilent | 0.103 | 2.5

## Open source implementations

These are community implementations with public source. No bitstream request is
needed -- the bitstream is built from the repository.

### Cmod A7-35T (Artix-7)

- Source: https://github.com/parthod0x/nexus-sk1024-cmod-a7
- Device: XC7A35T-1CPG236C, USB powered
- Measured **102.7 kH/s** at 25 MHz (MMCM from the 12 MHz board oscillator)
- 11,135 LUTs (53.5%), 12,689 FF (30.5%), 0 BRAM, 0 DSP, 0 DRC violations

Verified against this repository's own reference implementation: 280 nonces
returned by the board were recomputed with `nexus_keccak.cpp` and all 280
matched. Keccak, Threefish and full SK1024 each pass 8/8 known-answer vectors
generated from `nexus_skein.cpp` / `nexus_keccak.cpp` in simulation. It has run
live against a local wallet, logging "New block receipt acknowledged by FPGA".

At roughly 1/2000th of an FK33 this is not economically competitive. It exists
to give the hash channel an open, auditable reference implementation on a board
costing well under $100.
